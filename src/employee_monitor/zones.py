"""Desk-zone geometry, persistence, and calibration-time discovery."""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from employee_monitor.detection import Detection

PERSON_CLASS_ID = 0
SUPPORT_CLASS_IDS = frozenset((56, 62, 63, 66))
__all__ = [
    "AutoZoneCalibrator",
    "DeskZone",
    "assign_people_to_zones",
    "intersection_over_union",
    "load_zone_document",
    "load_zones",
    "match_person_to_zone",
    "merge_overlapping_zones",
    "person_zone_metrics",
    "phone_is_near_person",
    "save_zone_document",
    "scale_zones_to_frame",
]


@dataclass(frozen=True)
class DeskZone:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    origin: str = "auto"

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return float(self.x1), float(self.y1), float(self.x2), float(self.y2)


def load_zones(raw_zones: Iterable[Mapping[str, object]]) -> List[DeskZone]:
    zones: List[DeskZone] = []
    for index, raw in enumerate(raw_zones, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError("Each item in zones must be a mapping.")
        try:
            zone = DeskZone(
                name=str(raw.get("name", f"Desk {index}")),
                x1=int(raw["x1"]),
                y1=int(raw["y1"]),
                x2=int(raw["x2"]),
                y2=int(raw["y2"]),
                origin=str(raw.get("origin", "manual")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Zone {index} must have integer x1, y1, x2 and y2 coordinates."
            ) from exc
        if zone.x2 <= zone.x1 or zone.y2 <= zone.y1:
            raise ValueError(f"Zone {zone.name!r} must have positive width and height.")
        if any(existing.name == zone.name for existing in zones):
            raise ValueError(f"Zone names must be unique; {zone.name!r} is repeated.")
        zones.append(zone)
    return zones


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


def person_zone_metrics(
    person_box: Sequence[float], zone: DeskZone
) -> Tuple[bool, float]:
    center_x = (person_box[0] + person_box[2]) / 2.0
    center_y = (person_box[1] + person_box[3]) / 2.0
    center_inside = zone.x1 <= center_x <= zone.x2 and zone.y1 <= center_y <= zone.y2
    width = max(0.0, min(person_box[2], zone.x2) - max(person_box[0], zone.x1))
    height = max(0.0, min(person_box[3], zone.y2) - max(person_box[1], zone.y1))
    intersection = width * height
    person_area = max(0.0, person_box[2] - person_box[0]) * max(
        0.0, person_box[3] - person_box[1]
    )
    return center_inside, intersection / person_area if person_area else 0.0


def match_person_to_zone(
    person: Detection, zones: Sequence[DeskZone]
) -> Optional[DeskZone]:
    candidates = []
    for zone in zones:
        center_inside, overlap_ratio = person_zone_metrics(person.box, zone)
        if center_inside or overlap_ratio > 0.3:
            candidates.append((center_inside, overlap_ratio, zone))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def assign_people_to_zones(
    people: Iterable[Detection], zones: Sequence[DeskZone]
) -> Dict[str, Tuple[Detection, ...]]:
    assignments: Dict[str, List[Detection]] = {zone.name: [] for zone in zones}
    for person in people:
        winner = match_person_to_zone(person, zones)
        if winner is not None:
            assignments[winner.name].append(person)
    return {name: tuple(assigned) for name, assigned in assignments.items()}


def scale_zones_to_frame(
    zones: Sequence[DeskZone],
    reference_size: Optional[Tuple[int, int]],
    frame_size: Tuple[int, int],
) -> List[DeskZone]:
    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if reference_size is None:
        return list(zones)
    reference_width, reference_height = reference_size
    if reference_width <= 0 or reference_height <= 0:
        raise ValueError("zone_frame_size dimensions must be positive.")
    scale_x = frame_width / float(reference_width)
    scale_y = frame_height / float(reference_height)
    return [
        DeskZone(
            zone.name,
            int(round(zone.x1 * scale_x)),
            int(round(zone.y1 * scale_y)),
            int(round(zone.x2 * scale_x)),
            int(round(zone.y2 * scale_y)),
            zone.origin,
        )
        for zone in zones
    ]


def merge_overlapping_zones(
    zones: Sequence[DeskZone], iou_threshold: float = 0.4
) -> List[DeskZone]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1.")
    merged = list(zones)
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            for second_index in range(first_index + 1, len(merged)):
                first, second = merged[first_index], merged[second_index]
                if intersection_over_union(first.box, second.box) <= iou_threshold:
                    continue
                merged[first_index] = DeskZone(
                    first.name,
                    min(first.x1, second.x1),
                    min(first.y1, second.y1),
                    max(first.x2, second.x2),
                    max(first.y2, second.y2),
                    first.origin,
                )
                del merged[second_index]
                changed = True
                break
            if changed:
                break
    return [
        DeskZone(
            f"Desk {index}",
            zone.x1,
            zone.y1,
            zone.x2,
            zone.y2,
            zone.origin,
        )
        for index, zone in enumerate(merged, start=1)
    ]


def phone_is_near_person(
    phone_box: Sequence[float],
    person_box: Sequence[float],
    proximity_ratio: float,
) -> bool:
    width = max(1.0, person_box[2] - person_box[0])
    height = max(1.0, person_box[3] - person_box[1])
    phone_x = (phone_box[0] + phone_box[2]) / 2.0
    phone_y = (phone_box[1] + phone_box[3]) / 2.0
    reach_x, reach_y = width * proximity_ratio, height * proximity_ratio
    return (
        person_box[0] - reach_x <= phone_x <= person_box[2] + reach_x
        and person_box[1] - reach_y <= phone_y <= person_box[1] + height * 0.8 + reach_y
    )


def _zone_dict(zone: DeskZone) -> Dict[str, object]:
    return {
        "name": zone.name,
        "x1": zone.x1,
        "y1": zone.y1,
        "x2": zone.x2,
        "y2": zone.y2,
        "origin": zone.origin,
    }


def save_zone_document(
    path: Path,
    camera: str,
    zones: Sequence[DeskZone],
    frame_size: Tuple[int, int],
    origin: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "camera": camera,
        "origin": origin,
        "frame_size": {"width": frame_size[0], "height": frame_size[1]},
        "zones": [_zone_dict(zone) for zone in zones],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_zone_document(
    path: Path,
) -> Tuple[List[DeskZone], Optional[Tuple[int, int]], str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load zone file {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("zones"), list):
        raise ValueError(f"Zone file {path} must contain a zones list.")
    frame = document.get("frame_size")
    frame_size = None
    if frame is not None:
        if not isinstance(frame, dict):
            raise ValueError(f"frame_size in {path} must be a mapping.")
        width, height = frame.get("width"), frame.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError(f"frame_size in {path} must contain integer dimensions.")
        frame_size = (width, height)
    origin = str(document.get("origin", "auto"))
    zones = load_zones(document["zones"])
    return zones, frame_size, origin


class AutoZoneCalibrator:
    """Accumulate stable furniture anchors and stationary-person locations."""

    def __init__(
        self,
        cluster_distance_ratio: float = 0.1,
        still_seconds: float = 30.0,
        stationary_radius_ratio: float = 0.025,
    ) -> None:
        if cluster_distance_ratio <= 0:
            raise ValueError("cluster_distance_ratio must be positive.")
        if still_seconds <= 0 or stationary_radius_ratio <= 0:
            raise ValueError("Stillness thresholds must be positive.")
        self.cluster_distance_ratio = cluster_distance_ratio
        self.still_seconds = still_seconds
        self.stationary_radius_ratio = stationary_radius_ratio
        self._frame_size: Optional[Tuple[int, int]] = None
        self._anchors: List[Tuple[int, Tuple[float, float, float, float]]] = []
        self._persons: Dict[int, Dict[str, object]] = {}
        self._created_still_tracks = set()

    def observe(
        self,
        detections: Sequence[Detection],
        frame_size: Tuple[int, int],
        timestamp: float,
    ) -> List[DeskZone]:
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError("Frame dimensions must be positive.")
        if self._frame_size is None:
            self._frame_size = frame_size
        for detection in detections:
            if detection.class_id in SUPPORT_CLASS_IDS:
                self._add_anchor(detection)
            if detection.class_id == PERSON_CLASS_ID and detection.track_id is not None:
                self._observe_person(detection, timestamp)
        return self.stationary_zones()

    def _add_anchor(self, detection: Detection) -> None:
        if self._frame_size is None:
            return
        width, height = self._frame_size
        diagonal = math.hypot(width, height)
        center = _center(detection.box)
        tolerance = diagonal * 0.025
        for index, (class_id, box) in enumerate(self._anchors):
            if (
                class_id == detection.class_id
                and _distance(center, _center(box)) < tolerance
            ):
                self._anchors[index] = (
                    class_id,
                    tuple((old + new) / 2.0 for old, new in zip(box, detection.box)),
                )
                return
        self._anchors.append((detection.class_id, detection.box))

    def _observe_person(self, detection: Detection, timestamp: float) -> None:
        if self._frame_size is None:
            return
        track_id = detection.track_id
        center = _center(detection.box)
        tolerance = math.hypot(*self._frame_size) * self.stationary_radius_ratio
        state = self._persons.get(track_id)
        if state is None or _distance(center, state["anchor"]) > tolerance:
            self._persons[track_id] = {
                "anchor": center,
                "first_seen": timestamp,
                "box": detection.box,
            }
            return
        old_box = state["box"]
        state["box"] = (
            min(old_box[0], detection.box[0]),
            min(old_box[1], detection.box[1]),
            max(old_box[2], detection.box[2]),
            max(old_box[3], detection.box[3]),
        )
        if timestamp - float(state["first_seen"]) >= self.still_seconds:
            self._created_still_tracks.add(track_id)

    def stationary_zones(self) -> List[DeskZone]:
        zones = []
        for track_id in self._created_still_tracks:
            state = self._persons[track_id]
            zones.append(self._padded_zone(state["box"], f"Stationary {track_id}"))
        return merge_overlapping_zones(zones)

    def discover(self) -> List[DeskZone]:
        if self._frame_size is None:
            return []
        groups = self._anchor_clusters()
        zones = []
        for boxes in groups:
            bounds = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            zones.append(self._padded_zone(bounds, "Desk"))
        zones.extend(self.stationary_zones())
        return merge_overlapping_zones(zones)

    def _anchor_clusters(self) -> List[List[Tuple[float, float, float, float]]]:
        if self._frame_size is None or not self._anchors:
            return []
        width, height = self._frame_size
        threshold = self.cluster_distance_ratio
        remaining = set(range(len(self._anchors)))
        groups = []
        while remaining:
            seed = remaining.pop()
            group = {seed}
            stack = [seed]
            while stack:
                current = stack.pop()
                nearby = [
                    index
                    for index in remaining
                    if _normalized_distance(
                        _center(self._anchors[current][1]),
                        _center(self._anchors[index][1]),
                        width,
                        height,
                    )
                    <= threshold
                ]
                remaining.difference_update(nearby)
                group.update(nearby)
                stack.extend(nearby)
            groups.append([self._anchors[index][1] for index in group])
        return groups

    def _padded_zone(
        self, box: Sequence[float], name: str, origin: str = "auto"
    ) -> DeskZone:
        if self._frame_size is None:
            raise RuntimeError("Calibration has no frame dimensions.")
        width, height = self._frame_size
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        target_width = max(box_width * 1.8, width * 0.09)
        target_height = max(box_height * 1.6, height * 0.15)
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        x1 = max(0, int(round(center_x - target_width / 2)))
        y1 = max(0, int(round(center_y - target_height / 2)))
        x2 = min(width, int(round(center_x + target_width / 2)))
        y2 = min(height, int(round(center_y + target_height / 2)))
        return DeskZone(name, x1, y1, max(x1 + 1, x2), max(y1 + 1, y2), origin)


def _center(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _normalized_distance(
    first: Sequence[float], second: Sequence[float], width: int, height: int
) -> float:
    return math.hypot((first[0] - second[0]) / width, (first[1] - second[1]) / height)
