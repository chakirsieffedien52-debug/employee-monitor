"""Shared Ultralytics inference with optional pose and tiled phone detection."""

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from employee_monitor.config import ModelConfig

LOGGER = logging.getLogger(__name__)
PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67
SUPPORT_CLASS_IDS = (56, 62, 63, 66)
DETECTION_CLASS_IDS = (PERSON_CLASS_ID, 56, 62, 63, 66, PHONE_CLASS_ID)


@dataclass(frozen=True)
class Detection:
    box: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    track_id: Optional[int] = None
    keypoints: Optional[Tuple[Tuple[float, float, float], ...]] = None
    observed: bool = True


class Detector:
    def __init__(self, config: "ModelConfig") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install the project with `pip install -e .`."
            ) from exc

        self.config = config
        self._lock = threading.RLock()
        errors = []
        self.model = None
        for candidate in (config.name, config.fallback):
            if not candidate or (self.model is not None and candidate == config.name):
                continue
            try:
                self.model = YOLO(candidate)
                if candidate != config.name:
                    LOGGER.warning("Loaded fallback model %s", candidate)
                break
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
        if self.model is None:
            raise RuntimeError(
                "Could not load a YOLO model: {}".format("; ".join(errors))
            )

        self.device = "cpu"
        if config.use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    try:
                        self.model.to("cuda:0")
                        self.device = "cuda:0"
                        LOGGER.info("Using CUDA inference")
                    except RuntimeError as exc:
                        LOGGER.warning("CUDA initialization failed; using CPU: %s", exc)
                        self.model.to("cpu")
            except Exception as exc:
                LOGGER.warning("CUDA unavailable; using CPU inference: %s", exc)
                self.device = "cpu"

        self.pose_model = None
        if config.pose_enabled:
            try:
                self.pose_model = YOLO(config.pose_model)
                if self.device != "cpu":
                    try:
                        self.pose_model.to(self.device)
                    except RuntimeError as exc:
                        LOGGER.warning("CUDA pose setup failed; using CPU: %s", exc)
                        self._move_to_cpu()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load configured pose model {config.pose_model!r}: {exc}"
                ) from exc

    def detect(self, frame: np.ndarray, track: bool = False) -> List[Detection]:
        threshold = min(
            self.config.person_confidence,
            self.config.phone_confidence,
            self.config.object_confidence,
        )
        with self._lock:
            predict = self._track if track else self._predict
            results = predict(
                self.model,
                source=frame,
                classes=list(DETECTION_CLASS_IDS),
                conf=threshold,
                imgsz=self.config.imgsz,
                verbose=False,
                **({"persist": True} if track else {}),
            )
            detections = self._convert_results(results, offset=(0, 0))
            if self.config.tiled_phones:
                detections.extend(self._detect_phone_tiles(frame))
            detections = self._filter_and_deduplicate(detections)
            if self.pose_model is not None:
                detections = self._add_pose(frame, detections)
        return detections

    def _convert_results(
        self, results: Sequence[object], offset: Tuple[int, int]
    ) -> List[Detection]:
        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for index in range(len(boxes)):
                class_id = int(boxes.cls[index].item())
                confidence = float(boxes.conf[index].item())
                minimum = self._confidence_for(class_id)
                if minimum is None or confidence < minimum:
                    continue
                coords = boxes.xyxy[index].cpu().tolist()
                detections.append(
                    Detection(
                        box=(
                            float(coords[0] + offset[0]),
                            float(coords[1] + offset[1]),
                            float(coords[2] + offset[0]),
                            float(coords[3] + offset[1]),
                        ),
                        confidence=confidence,
                        class_id=class_id,
                    )
                )
        return detections

    def _confidence_for(self, class_id: int) -> Optional[float]:
        if class_id == PERSON_CLASS_ID:
            return self.config.person_confidence
        if class_id == PHONE_CLASS_ID:
            return self.config.phone_confidence
        if class_id in SUPPORT_CLASS_IDS:
            return self.config.object_confidence
        return None

    def _detect_phone_tiles(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        tile_size = self.config.tile_size
        step = max(1, int(tile_size * (1.0 - self.config.tile_overlap)))
        x_starts = _tile_starts(width, tile_size, step)
        y_starts = _tile_starts(height, tile_size, step)
        detections = []
        for y in y_starts:
            for x in x_starts:
                tile = frame[
                    y : min(y + tile_size, height), x : min(x + tile_size, width)
                ]
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    continue
                results = self._predict(
                    self.model,
                    source=tile,
                    classes=[PHONE_CLASS_ID],
                    conf=self.config.phone_confidence,
                    imgsz=max(tile.shape[0], tile.shape[1]),
                    verbose=False,
                )
                detections.extend(self._convert_results(results, (x, y)))
        return detections

    def _filter_and_deduplicate(
        self, detections: Sequence[Detection]
    ) -> List[Detection]:
        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        kept: List[Detection] = []
        for detection in ordered:
            if any(
                previous.class_id == detection.class_id
                and intersection_over_union(previous.box, detection.box) > 0.45
                for previous in kept
            ):
                continue
            kept.append(detection)
        return kept

    def _add_pose(
        self, frame: np.ndarray, detections: Sequence[Detection]
    ) -> List[Detection]:
        people = [item for item in detections if item.class_id == PERSON_CLASS_ID]
        if not people:
            return list(detections)
        results = self._predict(
            self.pose_model,
            source=frame,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        pose_boxes = []
        for result in results:
            keypoints = getattr(result, "keypoints", None)
            boxes = getattr(result, "boxes", None)
            if keypoints is None or boxes is None:
                continue
            xy = keypoints.xy.cpu().numpy()
            confidence = keypoints.conf
            scores = confidence.cpu().numpy() if confidence is not None else None
            for index, box in enumerate(boxes.xyxy.cpu().numpy()):
                points = []
                for point_index, (x, y) in enumerate(xy[index]):
                    score = (
                        float(scores[index][point_index]) if scores is not None else 1.0
                    )
                    points.append((float(x), float(y), score))
                pose_boxes.append((tuple(float(value) for value in box), tuple(points)))
        attached = {}
        for person in people:
            candidates = [
                (intersection_over_union(person.box, box), points)
                for box, points in pose_boxes
            ]
            if candidates:
                overlap, points = max(candidates, key=lambda item: item[0])
                if overlap > 0.1:
                    attached[id(person)] = points
        return [
            Detection(
                item.box,
                item.confidence,
                item.class_id,
                item.track_id,
                attached.get(id(item)),
                item.observed,
            )
            for item in detections
        ]

    def _predict(self, model, **kwargs):
        try:
            return model.predict(device=self.device, **kwargs)
        except RuntimeError as exc:
            if self.device == "cpu":
                raise
            LOGGER.warning("CUDA inference failed; retrying on CPU: %s", exc)
            self._move_to_cpu()
            return model.predict(device="cpu", **kwargs)

    def _track(self, model, **kwargs):
        try:
            return model.track(device=self.device, **kwargs)
        except RuntimeError as exc:
            if self.device == "cpu":
                raise
            LOGGER.warning("CUDA tracking failed; retrying on CPU: %s", exc)
            self._move_to_cpu()
            return model.track(device="cpu", **kwargs)

    def _move_to_cpu(self) -> None:
        self.model.to("cpu")
        if self.pose_model is not None:
            self.pose_model.to("cpu")
        self.device = "cpu"


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def _tile_starts(length: int, tile_size: int, step: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0
