"""Optional interactive manual zone override."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from employee_monitor.config import load_config, resolve_camera_source
from employee_monitor.zones import DeskZone, save_zone_document

LOGGER = logging.getLogger("employee_monitor.selector")


class ZoneSelector:
    def __init__(self) -> None:
        self.zones: List[DeskZone] = []
        self.start: Optional[Tuple[int, int]] = None
        self.current: Optional[Tuple[int, int, int, int]] = None
        self.drawing = False

    def on_mouse(
        self, event: int, x: int, y: int, flags: int, _parameter: object
    ) -> None:
        del flags
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.current = (x, y, x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and self.start is not None:
            self.current = (self.start[0], self.start[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing and self.start is not None:
            self.current = (self.start[0], self.start[1], x, y)
            self.commit_current()

    def commit_current(self) -> None:
        if self.current is None:
            return
        x1, y1, x2, y2 = self.current
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right > left and bottom > top:
            self.zones.append(
                DeskZone(
                    f"Desk {len(self.zones) + 1}",
                    left,
                    top,
                    right,
                    bottom,
                    "manual",
                )
            )
        self.current = None
        self.start = None
        self.drawing = False


def save_zones(
    config_path: Path,
    zones: List[DeskZone],
    frame_size: Tuple[int, int],
    camera_name: Optional[str] = None,
) -> Path:
    config = load_config(config_path)
    if camera_name is None:
        camera = config.cameras[0]
    else:
        camera = next(
            (item for item in config.cameras if item.name == camera_name), None
        )
        if camera is None:
            raise ValueError(f"Camera {camera_name!r} is not configured.")
    save_zone_document(
        camera.zones_file, camera.name, zones, frame_size, origin="manual"
    )
    return camera.zones_file


def run_zone_selector(config_path: Path, camera_name: Optional[str] = None) -> int:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    camera = (
        next((item for item in config.cameras if item.name == camera_name), None)
        if camera_name
        else config.cameras[0]
    )
    if camera is None:
        raise ValueError(f"Camera {camera_name!r} is not configured.")
    source = resolve_camera_source(camera.source, config_path.parent)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source for camera {camera.name!r}.")
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError("The camera opened but its first frame could not be read.")

    selector = ZoneSelector()
    window = "Select zones - u undo, s save, q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, selector.on_mouse)
    try:
        while True:
            preview = frame.copy()
            cv2.putText(
                preview,
                f"Camera {camera.name} - native frame {frame.shape[1]}x{frame.shape[0]}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for zone in selector.zones:
                cv2.rectangle(
                    preview, (zone.x1, zone.y1), (zone.x2, zone.y2), (0, 220, 0), 2
                )
                cv2.putText(
                    preview,
                    zone.name,
                    (zone.x1, max(20, zone.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
            if selector.current is not None:
                x1, y1, x2, y2 = selector.current
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("u"):
                if selector.current is not None:
                    selector.current = None
                    selector.start = None
                    selector.drawing = False
                elif selector.zones:
                    selector.zones.pop()
            elif key == ord("s"):
                selector.commit_current()
                if not selector.zones:
                    LOGGER.warning(
                        "Draw at least one valid desk rectangle before saving."
                    )
                    continue
                path = save_zones(
                    config_path,
                    selector.zones,
                    (frame.shape[1], frame.shape[0]),
                    camera.name,
                )
                LOGGER.info("Saved %d zones to %s", len(selector.zones), path)
                break
            elif key in (ord("q"), 27):
                LOGGER.info("Zone selection cancelled.")
                break
    finally:
        cv2.destroyAllWindows()
    return 0
