"""Command-line orchestration for multi-camera monitoring."""

import argparse
import logging
import math
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from employee_monitor.api import LiveState, create_app
from employee_monitor.capture import CameraWorker, FramePacket, test_camera
from employee_monitor.config import (
    AppConfig,
    CameraConfig,
    ConfigurationError,
    Source,
    load_config,
    resolve_camera_source,
)
from employee_monitor.desk_memory import (
    AbsentDesk,
    DeskMemory,
    send_urgent_alert,
)
from employee_monitor.detection import (
    PERSON_CLASS_ID,
    PHONE_CLASS_ID,
    Detection,
    Detector,
    apply_clahe,
)
from employee_monitor.privacy import FaceBlur
from employee_monitor.rendering import (
    compose_grid,
    draw_detection_overlay,
    draw_overlay,
)
from employee_monitor.reporting import EventLogger, ReportExporter, read_daily_events
from employee_monitor.status import (
    ABSENT,
    PHONE_USAGE,
    WORKING,
    PersonStatusTracker,
    StatusTracker,
    assign_person_phone_statuses,
)
from employee_monitor.tracking import PersonTracker
from employee_monitor.zone_selector import run_zone_selector
from employee_monitor.zones import (
    AutoZoneCalibrator,
    DeskZone,
    assign_people_to_zones,
    load_zone_document,
    merge_overlapping_zones,
    phone_is_near_person,
    save_zone_document,
    scale_zones_to_frame,
)

LOGGER = logging.getLogger("employee_monitor")


@dataclass
class CameraRuntime:
    config: CameraConfig
    worker: CameraWorker
    event_logger: EventLogger
    calibrator: AutoZoneCalibrator
    person_tracker: PersonTracker
    face_blur: FaceBlur
    zones: List[DeskZone] = field(default_factory=list)
    frame_size: Optional[Tuple[int, int]] = None
    status_tracker: Optional[StatusTracker] = None
    person_status_tracker: Optional[PersonStatusTracker] = None
    desk_memory: Optional[DeskMemory] = None
    calibration_started: Optional[float] = None
    calibration_finished: bool = False
    processed_frames: int = 0
    previous_index: Optional[int] = None
    previous_processing_time: Optional[float] = None
    memory_time_origin: Optional[float] = None
    fps: float = 0.0
    last_frame: Optional[np.ndarray] = None
    last_output: Optional[np.ndarray] = None
    last_people: Tuple[Detection, ...] = ()
    last_phones: Tuple[Detection, ...] = ()
    people_by_zone: Dict[str, Tuple[Detection, ...]] = field(default_factory=dict)
    writer: Optional[cv2.VideoWriter] = None
    had_frame: bool = False


def _phone_use(person: Detection, phones: Sequence[Detection], ratio: float) -> bool:
    for phone in phones:
        if phone_is_near_person(phone.box, person.box, ratio):
            return True
        if person.keypoints:
            phone_x = (phone.box[0] + phone.box[2]) / 2.0
            phone_y = (phone.box[1] + phone.box[3]) / 2.0
            wrists = [
                person.keypoints[index]
                for index in (9, 10)
                if index < len(person.keypoints)
            ]
            max_wrist_distance = max(24.0, (person.box[3] - person.box[1]) * 0.15)
            if any(
                score >= 0.2
                and math.hypot(x - phone_x, y - phone_y) < max_wrist_distance
                for x, y, score in wrists
            ):
                return True
            if _head_is_down(person.keypoints) and _phone_near_torso(phone, person.box):
                return True
    return False


def _head_is_down(keypoints: Sequence[Tuple[float, float, float]]) -> bool:
    if len(keypoints) < 7 or keypoints[0][2] < 0.2:
        return False
    shoulders = [keypoints[index] for index in (5, 6) if keypoints[index][2] >= 0.2]
    return bool(shoulders) and keypoints[0][1] > sum(
        point[1] for point in shoulders
    ) / len(shoulders)


def _phone_near_torso(phone: Detection, person_box: Sequence[float]) -> bool:
    center_x = (phone.box[0] + phone.box[2]) / 2.0
    center_y = (phone.box[1] + phone.box[3]) / 2.0
    width = person_box[2] - person_box[0]
    height = person_box[3] - person_box[1]
    return (
        person_box[0] - width * 0.15 <= center_x <= person_box[2] + width * 0.15
        and person_box[1] + height * 0.2 <= center_y <= person_box[1] + height * 0.85
    )


def _build_observations(
    zones: Sequence[DeskZone],
    people: Sequence[Detection],
    phones: Sequence[Detection],
    proximity_ratio: float,
) -> Tuple[Dict[str, str], Dict[str, Tuple[Detection, ...]]]:
    people_by_zone = assign_people_to_zones(people, zones)
    observations = {}
    for zone in zones:
        assigned = people_by_zone[zone.name]
        observations[zone.name] = (
            ABSENT
            if not assigned
            else (
                PHONE_USAGE
                if any(
                    _phone_use(person, phones, proximity_ratio) for person in assigned
                )
                else WORKING
            )
        )
    return observations, people_by_zone


def _camera_runtime(
    config: AppConfig,
    camera: CameraConfig,
    recalibrate: bool,
    event_logger: Optional[EventLogger] = None,
    desk_memory_enabled: bool = True,
) -> CameraRuntime:
    source = resolve_camera_source(camera.source, config.config_path.parent)
    worker = CameraWorker(
        camera.name,
        source,
        config.reconnect_initial_seconds,
        config.reconnect_max_seconds,
    )
    zones = []
    frame_size = None
    calibration_finished = False
    if config.mode == "detect":
        calibration_finished = True
    else:
        if not recalibrate and camera.zones_file.is_file():
            zones, frame_size, origin = load_zone_document(camera.zones_file)
            calibration_finished = True
            LOGGER.info(
                "Loaded %d %s zone(s) for camera %s",
                len(zones),
                origin,
                camera.name,
            )
        elif not config.auto_discovery:
            if not camera.manual_zones:
                raise ConfigurationError(
                    f"Camera {camera.name!r} has no saved zones and auto_discovery is disabled."
                )
            zones = list(camera.manual_zones)
            frame_size = camera.zone_frame_size
            calibration_finished = True
    face_blur = FaceBlur(config.blur_faces)
    if event_logger is None:
        event_logger = EventLogger(config.logs_dir / camera.name, camera.name)
    return CameraRuntime(
        config=camera,
        worker=worker,
        event_logger=event_logger,
        calibrator=AutoZoneCalibrator(
            config.cluster_distance_ratio, config.status.still_seconds
        ),
        person_tracker=PersonTracker(config.status.persistence_frames),
        face_blur=face_blur,
        desk_memory=(
            DeskMemory(
                camera.desks_file,
                min_still_seconds=config.desks_memory.min_still_seconds,
                absent_after_seconds=config.desks_memory.absent_after_seconds,
                urgent_after_seconds=config.desks_memory.urgent_after_seconds,
                margin=config.desks_memory.margin,
                max_desks=config.desks_memory.max_desks,
                forget_after_minutes=config.desks_memory.forget_after_minutes,
            )
            if config.mode == "detect" and desk_memory_enabled
            else None
        ),
        zones=zones,
        frame_size=frame_size,
        calibration_finished=calibration_finished,
    )


def _initialize_frame_size(
    runtime: CameraRuntime, frame: np.ndarray, config: AppConfig, fps: float
) -> None:
    actual_size = (frame.shape[1], frame.shape[0])
    reference_size = runtime.frame_size
    runtime.frame_size = actual_size
    if runtime.zones:
        runtime.zones = scale_zones_to_frame(runtime.zones, reference_size, actual_size)
        _configure_status(runtime, config, fps)


def _configure_status(runtime: CameraRuntime, config: AppConfig, fps: float) -> None:
    if not runtime.zones:
        return
    runtime.status_tracker = StatusTracker(
        [zone.name for zone in runtime.zones],
        stable_frames=config.status.stable_frames,
        fps=fps if math.isfinite(fps) and fps > 0 else 30.0,
    )


def _finish_calibration(runtime: CameraRuntime, config: AppConfig) -> None:
    if runtime.calibration_finished:
        return
    if runtime.frame_size is None:
        raise RuntimeError("Cannot calibrate before a camera frame is received.")
    zones = runtime.calibrator.discover() if config.auto_discovery else []
    if not zones and runtime.config.manual_zones:
        zones = list(runtime.config.manual_zones)
        zones = scale_zones_to_frame(
            zones, runtime.config.zone_frame_size, runtime.frame_size
        )
        LOGGER.warning(
            "Auto-discovery found no furniture for %s; using configured manual zones",
            runtime.config.name,
        )
    if not zones:
        raise RuntimeError(
            f"No desk zones discovered for camera {runtime.config.name!r}. Run "
            "`employee-monitor select-zones` or check "
            "camera framing and model detections."
        )
    runtime.zones = merge_overlapping_zones(zones)
    if runtime.frame_size is None:
        raise RuntimeError("Cannot save zones before a camera frame is received.")
    save_zone_document(
        runtime.config.zones_file,
        runtime.config.name,
        runtime.zones,
        runtime.frame_size,
        runtime.zones[0].origin,
    )
    runtime.calibration_finished = True
    _configure_status(runtime, config, runtime.worker.fps)
    LOGGER.info(
        "Calibration completed for %s: %d desk zone(s) saved to %s",
        runtime.config.name,
        len(runtime.zones),
        runtime.config.zones_file,
    )


def _update_stationary_zones(
    runtime: CameraRuntime,
    config: AppConfig,
    detections: Sequence[Detection],
    timestamp: float,
) -> None:
    if runtime.frame_size is None:
        raise RuntimeError("Cannot update stationary zones before receiving a frame.")
    new_stationary = runtime.calibrator.observe(
        [
            item
            for item in detections
            if item.observed or item.class_id != PERSON_CLASS_ID
        ],
        runtime.frame_size,
        timestamp,
    )
    if not runtime.calibration_finished or not new_stationary:
        return
    candidate = merge_overlapping_zones(runtime.zones + new_stationary)
    if [(z.x1, z.y1, z.x2, z.y2) for z in candidate] == [
        (z.x1, z.y1, z.x2, z.y2) for z in runtime.zones
    ]:
        return
    runtime.zones = candidate
    if runtime.status_tracker is not None:
        runtime.status_tracker.add_desks(zone.name for zone in candidate)
    save_zone_document(
        runtime.config.zones_file,
        runtime.config.name,
        runtime.zones,
        runtime.frame_size,
        "auto",
    )
    LOGGER.info("Added stationary-person zone for camera %s", runtime.config.name)


def _durations(runtime: CameraRuntime) -> Dict[str, Dict[str, float]]:
    if runtime.status_tracker is None:
        return {}
    result: Dict[str, Dict[str, float]] = {}
    for desk, status, seconds in runtime.status_tracker.summary():
        result.setdefault(desk, {})[status] = seconds
    return result


def _desk_memory_timestamp(runtime: CameraRuntime, packet: FramePacket) -> float:
    if runtime.worker.is_live:
        return time.time()
    fps = packet.fps if math.isfinite(packet.fps) and packet.fps > 0 else 30.0
    video_time = packet.index / fps
    if runtime.memory_time_origin is None:
        runtime.memory_time_origin = time.time() - video_time
    return runtime.memory_time_origin + video_time


def _process_packet(
    runtime: CameraRuntime,
    packet: FramePacket,
    detector: Detector,
    config: AppConfig,
    live_state: LiveState,
) -> None:
    if not runtime.had_frame:
        _initialize_frame_size(runtime, packet.frame, config, packet.fps)
        runtime.had_frame = True
    current_time = time.monotonic()
    if runtime.calibration_started is None:
        runtime.calibration_started = packet.captured_at
    runtime.processed_frames += 1
    interval = (
        max(1, packet.index - runtime.previous_index)
        if runtime.previous_index is not None
        else 1
    )
    inference_frame = (
        apply_clahe(packet.frame) if config.clahe_enabled else packet.frame
    )
    detections = detector.detect(inference_frame, track=config.mode == "detect")
    tracked_detections = runtime.person_tracker.update(detections)
    people = tuple(
        item for item in tracked_detections if item.class_id == PERSON_CLASS_ID
    )
    phones = tuple(
        item for item in tracked_detections if item.class_id == PHONE_CLASS_ID
    )

    if config.debug:
        LOGGER.debug(
            "%s frame=%d people=%s phones=%s",
            runtime.config.name,
            packet.index,
            [
                (item.track_id, item.box, item.confidence, item.observed)
                for item in people
            ],
            [(item.box, item.confidence) for item in phones],
        )

    if config.mode == "desks" and not runtime.calibration_finished:
        _update_stationary_zones(
            runtime, config, tracked_detections, packet.captured_at
        )
        elapsed = packet.captured_at - runtime.calibration_started
        if elapsed >= runtime.config.calibration_seconds:
            _finish_calibration(runtime, config)
    elif config.mode == "desks" and config.auto_discovery:
        _update_stationary_zones(
            runtime, config, tracked_detections, packet.captured_at
        )

    if config.mode == "detect":
        if runtime.person_status_tracker is None:
            runtime.person_status_tracker = PersonStatusTracker(
                config.status.stable_frames
            )
        observations = assign_person_phone_statuses(
            people, phones, config.status.phone_proximity_ratio
        )
        person_statuses = runtime.person_status_tracker
        changes = person_statuses.update(
            observations,
            (person.track_id for person in people if person.track_id is not None),
        )
        for change in changes:
            runtime.event_logger.log_change(
                change.desk, change.status, camera=runtime.config.name
            )
        statuses = person_statuses.statuses
        runtime.people_by_zone = {}
    elif runtime.status_tracker is not None and runtime.zones:
        observations, people_by_zone = _build_observations(
            runtime.zones,
            people,
            phones,
            config.status.phone_proximity_ratio,
        )
        changes = runtime.status_tracker.update(observations, elapsed_frames=interval)
        for change in changes:
            runtime.event_logger.log_change(change.desk, change.status)
        runtime.people_by_zone = people_by_zone
        statuses = runtime.status_tracker.statuses
    else:
        statuses = {}

    if runtime.previous_processing_time is not None:
        elapsed_processing = current_time - runtime.previous_processing_time
        if elapsed_processing > 0:
            runtime.fps = 1.0 / elapsed_processing
    runtime.previous_processing_time = current_time
    runtime.previous_index = packet.index
    runtime.last_people, runtime.last_phones = people, phones
    runtime.last_frame = packet.frame
    output_frame = runtime.face_blur.apply(packet.frame)
    if config.mode == "detect":
        absent_desks: Tuple[AbsentDesk, ...] = ()
        urgent_desk = None
        if runtime.desk_memory is not None:
            memory_time = _desk_memory_timestamp(runtime, packet)
            memory_events = runtime.desk_memory.update(
                tracked_detections,
                memory_time,
                (packet.frame.shape[1], packet.frame.shape[0]),
            )
            for event in memory_events:
                runtime.event_logger.log_change(
                    event.desk,
                    event.status,
                    camera=runtime.config.name,
                    duration_seconds=event.duration_seconds,
                )
                if event.status == "URGENT":
                    send_urgent_alert(
                        runtime.config.name,
                        event.desk,
                        event.duration_seconds,
                        config.desks_memory.webhook_url,
                    )
            absent_desks = runtime.desk_memory.absent_desks(memory_time)
            urgent_desk = next((desk for desk in absent_desks if desk.urgent), None)
            statuses = dict(statuses)
            statuses.update({desk.name: ABSENT for desk in absent_desks})
        runtime.last_output = draw_detection_overlay(
            output_frame,
            people,
            phones,
            statuses,
            runtime.fps,
            absent_desks=absent_desks,
            urgent_message=(
                f"URGENT: {urgent_desk.name} ABSENT" if urgent_desk else None
            ),
            blink=int(time.monotonic() * 2) % 2 == 0,
        )
    else:
        runtime.last_output = draw_overlay(
            output_frame,
            runtime.zones,
            statuses,
            people,
            phones,
            runtime.people_by_zone,
            runtime.fps,
        )

    if config.save_video or not runtime.worker.is_live:
        if runtime.writer is None:
            height, width = runtime.last_output.shape[:2]
            camera_output = config.output_dir / runtime.config.name / "result.mp4"
            camera_output.parent.mkdir(parents=True, exist_ok=True)
            fps = packet.fps if packet.fps > 0 else 30.0
            runtime.writer = cv2.VideoWriter(
                str(camera_output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not runtime.writer.isOpened():
                runtime.writer.release()
                runtime.writer = None
                raise RuntimeError(f"Could not create output video: {camera_output}")
        runtime.writer.write(runtime.last_output)

    live_state.update(
        runtime.config.name,
        statuses,
        _durations(runtime),
        runtime.fps,
        not runtime.worker.finished,
    )


def _run_dashboard(config: AppConfig, live_state: LiveState, events_by_camera: dict):
    import uvicorn

    report_exporter = ReportExporter(config.logs_dir / "reports")
    app = create_app(live_state, report_exporter, events_by_camera)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.dashboard_host,
            port=config.dashboard_port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, name="employee-dashboard", daemon=True)
    thread.start()
    LOGGER.info(
        "Dashboard available at http://%s:%d",
        config.dashboard_host,
        config.dashboard_port,
    )
    return server, thread


def run(
    config_path: Path,
    headless: bool = False,
    grid: bool = False,
    recalibrate: bool = False,
    max_frames: Optional[int] = None,
    dashboard: bool = False,
    clear_desks: bool = False,
    no_desk_memory: bool = False,
    show: bool = False,
) -> int:
    config = load_config(config_path)
    if clear_desks:
        for camera in config.cameras:
            if camera.desks_file.exists():
                camera.desks_file.unlink()
    for camera in config.cameras:
        resolve_camera_source(camera.source, config.config_path.parent)
    detector = Detector(config.model)
    runtimes = []
    detect_logger = EventLogger(config.logs_dir) if config.mode == "detect" else None
    desk_memory_enabled = (
        config.mode == "detect" and config.desks_memory.enabled and not no_desk_memory
    )
    try:
        for camera in config.cameras:
            runtimes.append(
                _camera_runtime(
                    config,
                    camera,
                    recalibrate,
                    detect_logger,
                    desk_memory_enabled,
                )
            )
    except Exception:
        for runtime in runtimes:
            runtime.event_logger.close()
        if detect_logger is not None:
            detect_logger.close()
        raise
    live_state = LiveState()
    events_by_camera: Dict[str, List[dict]] = {
        runtime.config.name: [] for runtime in runtimes
    }
    server_info = None
    error = None
    stopped_by_escape = False
    try:
        if dashboard:
            server_info = _run_dashboard(config, live_state, events_by_camera)
        if not headless:
            cv2.namedWindow("Employee Monitor", cv2.WINDOW_NORMAL)
        for runtime in runtimes:
            runtime.worker.start()
        LOGGER.info(
            "Monitoring %d camera(s) with %s on %s",
            len(runtimes),
            config.model.name,
            detector.device,
        )
        paused = False
        while True:
            processed = False
            all_finished = True
            latest_outputs = {}
            for runtime in runtimes:
                if max_frames is not None and runtime.processed_frames >= max_frames:
                    runtime.worker.stop()
                    if (
                        config.mode == "desks"
                        and runtime.had_frame
                        and not runtime.calibration_finished
                    ):
                        _finish_calibration(runtime, config)
                    if not runtime.worker.finished:
                        all_finished = False
                    if runtime.last_output is not None:
                        latest_outputs[runtime.config.name] = runtime.last_output
                    continue
                packet = runtime.worker.get_latest(timeout=0)
                if packet is not None:
                    processed = True
                    _process_packet(runtime, packet, detector, config, live_state)
                elif (
                    config.mode == "desks"
                    and runtime.worker.finished
                    and not runtime.calibration_finished
                ):
                    if runtime.had_frame:
                        _finish_calibration(runtime, config)
                if not runtime.worker.finished:
                    all_finished = False
                if runtime.last_output is not None:
                    latest_outputs[runtime.config.name] = runtime.last_output
                if max_frames is not None and runtime.processed_frames >= max_frames:
                    runtime.worker.stop()
            if processed:
                for runtime in runtimes:
                    event_dir = (
                        config.logs_dir
                        if config.mode == "detect"
                        else config.logs_dir / runtime.config.name
                    )
                    camera_events = read_daily_events(event_dir)
                    events_by_camera[runtime.config.name] = (
                        [
                            event
                            for event in camera_events
                            if event.get("camera") == runtime.config.name
                        ]
                        if config.mode == "detect"
                        else camera_events
                    )
            if all_finished and not processed:
                break

            if not headless:
                if grid:
                    if latest_outputs:
                        cv2.imshow("Employee Monitor", compose_grid(latest_outputs))
                else:
                    for name, frame in latest_outputs.items():
                        cv2.imshow(name, frame)
                key = cv2.waitKey(1 if not paused else 50) & 0xFF
                if key == 27:
                    stopped_by_escape = True
                    break
                if key == ord(" "):
                    paused = not paused

            if not processed:
                time.sleep(0.01)
        if not any(runtime.had_frame for runtime in runtimes):
            raise RuntimeError("No frames were received from any configured camera.")
        if show and not headless and not stopped_by_escape:
            while True:
                if grid:
                    outputs = {
                        runtime.config.name: runtime.last_output
                        for runtime in runtimes
                        if runtime.last_output is not None
                    }
                    if outputs:
                        cv2.imshow("Employee Monitor", compose_grid(outputs))
                else:
                    for runtime in runtimes:
                        if runtime.last_output is not None:
                            cv2.imshow(runtime.config.name, runtime.last_output)
                if cv2.waitKey(30) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        LOGGER.info("Stopping monitoring after keyboard interrupt")
    except Exception as exc:
        error = exc
    finally:
        for runtime in runtimes:
            runtime.worker.stop()
            if runtime.writer is not None:
                runtime.writer.release()
            if config.mode == "desks":
                runtime.event_logger.save_summary(
                    runtime.status_tracker.summary()
                    if runtime.status_tracker is not None
                    else []
                )
            runtime.event_logger.close()
            event_dir = (
                config.logs_dir
                if config.mode == "detect"
                else config.logs_dir / runtime.config.name
            )
            camera_events = read_daily_events(event_dir)
            events_by_camera[runtime.config.name] = (
                [
                    event
                    for event in camera_events
                    if event.get("camera") == runtime.config.name
                ]
                if config.mode == "detect"
                else camera_events
            )
        if config.mode == "detect" and detect_logger is not None:
            detect_logger.save_summary([])
        if not headless:
            cv2.destroyAllWindows()
        if server_info is not None:
            server, thread = server_info
            server.should_exit = True
            thread.join(timeout=5.0)
        exporter = ReportExporter(config.logs_dir / "reports")
        exporter.export_daily(
            events_by_camera,
            {runtime.config.name: _durations(runtime) for runtime in runtimes},
        )
        processed_frames = sum(runtime.processed_frames for runtime in runtimes)
        event_count = sum(len(events) for events in events_by_camera.values())
        summary_paths = (
            [config.logs_dir / "summary.csv"]
            if config.mode == "detect"
            else [
                config.logs_dir / runtime.config.name / "summary.csv"
                for runtime in runtimes
            ]
        )
        print(
            f"Run summary: {processed_frames} frame(s), {event_count} event(s). "
            f"Summary: {', '.join(str(path) for path in summary_paths)}"
        )
    if error is not None:
        raise error
    return 0


def _parse_cli_source(value: str) -> Source:
    return int(value) if value.isdigit() else value


def _redact_source(source: Source) -> str:
    if isinstance(source, int):
        return str(source)
    parsed = urlsplit(source)
    if not parsed.scheme or not parsed.hostname:
        return source
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _select_zones_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="employee-monitor select-zones",
        description="Draw a manual camera-zone override.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.example.yaml",
        help="Configuration YAML.",
    )
    parser.add_argument("--camera", help="Configured camera name.")
    args = parser.parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve()
        config = load_config(config_path)
        logging.basicConfig(
            level=logging.DEBUG if config.debug else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        return run_zone_selector(config_path, args.camera)
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception:
        LOGGER.exception("Unexpected zone-selection error")
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "select-zones":
        return _select_zones_command(arguments[1:])
    if arguments and arguments[0] == "run":
        arguments = arguments[1:]
    parser = argparse.ArgumentParser(
        prog="employee-monitor",
        usage="employee-monitor [run] [options]",
        description="Local-first multi-camera desk activity monitor.",
        epilog="Use 'employee-monitor select-zones' for optional manual desk zones.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.example.yaml",
        help="Configuration YAML.",
    )
    parser.add_argument("--headless", action="store_true", help="Do not open windows.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Keep the final live preview open until Esc is pressed.",
    )
    parser.add_argument(
        "--clear-desks",
        action="store_true",
        help="Delete remembered desk JSON files before starting.",
    )
    parser.add_argument(
        "--no-desk-memory",
        action="store_true",
        help="Disable remembered-desk learning and absence alerts.",
    )
    parser.add_argument("--grid", action="store_true", help="Show cameras in one grid.")
    parser.add_argument(
        "--recalibrate", action="store_true", help="Rediscover desk zones."
    )
    parser.add_argument(
        "--test-camera",
        metavar="SOURCE",
        help="Print source resolution and FPS, then exit.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many processed frames per camera (useful for smoke tests).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Serve the live dashboard on the configured port.",
    )
    parser.add_argument("--version", action="version", version="employee-monitor 0.2.0")
    args = parser.parse_args(arguments)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive.")
    try:
        config_path = Path(args.config).expanduser().resolve()
        config = load_config(config_path)
        logging.basicConfig(
            level=logging.DEBUG if config.debug else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        if args.test_camera is not None:
            source = resolve_camera_source(
                _parse_cli_source(args.test_camera), config_path.parent
            )
            result = test_camera(source)
            if result is None:
                raise RuntimeError(
                    f"Could not read a frame from camera source {source!r}."
                )
            LOGGER.info(
                "Camera source %s: %dx%d at %.2f FPS",
                _redact_source(source),
                result["width"],
                result["height"],
                result["fps"],
            )
            return 0
        return run(
            config_path,
            headless=args.headless,
            show=args.show,
            grid=args.grid,
            recalibrate=args.recalibrate,
            max_frames=args.max_frames,
            dashboard=args.dashboard,
            clear_desks=args.clear_desks,
            no_desk_memory=args.no_desk_memory,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        LOGGER.error("%s", exc)
        return 1
    except Exception:
        LOGGER.exception("Unexpected monitoring error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
