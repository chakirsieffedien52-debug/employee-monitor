"""Remember stable work areas and track their occupancy."""

import json
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

from employee_monitor.detection import PERSON_CLASS_ID, Detection
from employee_monitor.zones import SUPPORT_CLASS_IDS, intersection_over_union

LOGGER = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class AbsentDesk:
    name: str
    box: Box
    absent_seconds: float
    urgent: bool


@dataclass(frozen=True)
class DeskMemoryEvent:
    desk: str
    status: str
    duration_seconds: float


@dataclass
class _RememberedDesk:
    name: str
    box: Box
    created_at: float
    last_occupied_at: float
    absence_alerted: bool = False
    urgent_alerted: bool = False


@dataclass
class _StillCandidate:
    started_at: float
    center: Tuple[float, float]
    box: Tuple[float, float, float, float]
    remembered: bool = False


class DeskMemory:
    def __init__(
        self,
        path: Path,
        min_still_seconds: float = 5.0,
        absent_after_seconds: float = 3.0,
        urgent_after_seconds: float = 30.0,
        margin: float = 0.25,
        max_desks: int = 12,
        forget_after_minutes: float = 0.0,
    ) -> None:
        if min_still_seconds < 0 or absent_after_seconds < 0:
            raise ValueError("Desk memory time thresholds cannot be negative.")
        if urgent_after_seconds < absent_after_seconds:
            raise ValueError(
                "urgent_after_seconds cannot precede absent_after_seconds."
            )
        if margin < 0 or max_desks < 1 or forget_after_minutes < 0:
            raise ValueError("Invalid desk memory margin, limit, or expiry.")
        self.path = path
        self.min_still_seconds = min_still_seconds
        self.absent_after_seconds = absent_after_seconds
        self.urgent_after_seconds = urgent_after_seconds
        self.margin = margin
        self.max_desks = max_desks
        self.forget_after_minutes = forget_after_minutes
        self._desks: Dict[str, _RememberedDesk] = {}
        self._candidates: Dict[int, _StillCandidate] = {}
        self._next_id = 1
        self._last_saved_at = float("-inf")
        self._dirty = False
        self._lock = threading.RLock()
        self._load()

    @property
    def desks(self) -> Tuple[Tuple[str, Box], ...]:
        with self._lock:
            return tuple(
                (desk.name, desk.box)
                for desk in sorted(self._desks.values(), key=_desk_number)
            )

    def update(
        self,
        detections: Sequence[Detection],
        timestamp: float,
        frame_size: Tuple[int, int],
    ) -> Tuple[DeskMemoryEvent, ...]:
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError("Frame dimensions must be positive.")
        with self._lock:
            events: List[DeskMemoryEvent] = []
            people = [item for item in detections if item.class_id == PERSON_CLASS_ID]
            observed_people = [
                item for item in people if item.observed and item.track_id is not None
            ]
            visible_ids = {
                item.track_id for item in people if item.track_id is not None
            }
            self._candidates = {
                track_id: candidate
                for track_id, candidate in self._candidates.items()
                if track_id in visible_ids
            }

            for person in observed_people:
                track_id = person.track_id
                center = _center(person.box)
                candidate = self._candidates.get(track_id)
                if candidate is None or _distance(center, candidate.center) > max(
                    12.0, (person.box[3] - person.box[1]) * 0.12
                ):
                    candidate = _StillCandidate(timestamp, center, person.box)
                    self._candidates[track_id] = candidate
                else:
                    candidate.box = _union_float(candidate.box, person.box)
                if (
                    not candidate.remembered
                    and timestamp - candidate.started_at >= self.min_still_seconds
                ):
                    desk_box = self._candidate_box(
                        candidate.box, detections, frame_size
                    )
                    memory_events, candidate.remembered = self._remember(
                        desk_box, timestamp
                    )
                    events.extend(memory_events)

            occupied_names = {
                desk.name
                for desk in self._desks.values()
                if any(_person_inside(person.box, desk.box) for person in people)
            }
            forget_after_seconds = self.forget_after_minutes * 60.0
            for desk in list(self._desks.values()):
                absence_seconds = max(0.0, timestamp - desk.last_occupied_at)
                if desk.name in occupied_names:
                    if desk.absence_alerted:
                        events.append(
                            DeskMemoryEvent(desk.name, "ABSENT END", absence_seconds)
                        )
                    desk.last_occupied_at = timestamp
                    desk.absence_alerted = False
                    desk.urgent_alerted = False
                    continue

                if (
                    not desk.absence_alerted
                    and absence_seconds >= self.absent_after_seconds
                ):
                    desk.absence_alerted = True
                    events.append(
                        DeskMemoryEvent(desk.name, "ABSENT START", absence_seconds)
                    )
                if (
                    desk.absence_alerted
                    and not desk.urgent_alerted
                    and absence_seconds >= self.urgent_after_seconds
                ):
                    desk.urgent_alerted = True
                    events.append(DeskMemoryEvent(desk.name, "URGENT", absence_seconds))
                if forget_after_seconds > 0 and absence_seconds >= forget_after_seconds:
                    if desk.absence_alerted:
                        events.append(
                            DeskMemoryEvent(desk.name, "ABSENT END", absence_seconds)
                        )
                    del self._desks[desk.name]
                    self._dirty = True

            if events or self._dirty or timestamp - self._last_saved_at >= 1.0:
                self._save(timestamp)
            return tuple(events)

    def absent_desks(self, timestamp: float) -> Tuple[AbsentDesk, ...]:
        with self._lock:
            absent = []
            for desk in sorted(self._desks.values(), key=_desk_number):
                duration = max(0.0, timestamp - desk.last_occupied_at)
                if duration < self.absent_after_seconds:
                    continue
                absent.append(
                    AbsentDesk(
                        desk.name,
                        desk.box,
                        duration,
                        duration >= self.urgent_after_seconds,
                    )
                )
            return tuple(absent)

    def _candidate_box(
        self,
        accumulated_person_box: Sequence[float],
        detections: Sequence[Detection],
        frame_size: Tuple[int, int],
    ) -> Box:
        person_center = _center(accumulated_person_box)
        person_width = max(1.0, accumulated_person_box[2] - accumulated_person_box[0])
        person_height = max(1.0, accumulated_person_box[3] - accumulated_person_box[1])
        bounds = tuple(accumulated_person_box)
        for item in detections:
            if (
                item.class_id not in SUPPORT_CLASS_IDS
                or not item.observed
                or _normalized_distance(
                    item.box, person_center, person_width, person_height
                )
                > 1.25
            ):
                continue
            bounds = _union_float(bounds, item.box)

        margin_x = person_width * self.margin
        margin_y = person_height * self.margin
        expanded = (
            bounds[0] - margin_x,
            bounds[1] - margin_y,
            bounds[2] + margin_x,
            bounds[3] + margin_y,
        )
        width, height = frame_size
        return (
            max(0, int(math.floor(expanded[0]))),
            max(0, int(math.floor(expanded[1]))),
            min(width, int(math.ceil(expanded[2]))),
            min(height, int(math.ceil(expanded[3]))),
        )

    def _remember(
        self, box: Box, timestamp: float
    ) -> Tuple[List[DeskMemoryEvent], bool]:
        events = []
        overlapping = []
        merged_box = box
        while True:
            newly_overlapping = [
                desk
                for desk in self._desks.values()
                if desk not in overlapping
                and intersection_over_union(desk.box, merged_box) > 0.4
            ]
            if not newly_overlapping:
                break
            overlapping.extend(newly_overlapping)
            for desk in newly_overlapping:
                merged_box = _union_int(merged_box, desk.box)
        if not overlapping and len(self._desks) >= self.max_desks:
            return events, False
        if not overlapping:
            name = f"Desk {self._next_id}"
            self._next_id += 1
            self._desks[name] = _RememberedDesk(name, box, timestamp, timestamp)
            self._dirty = True
            return events, True

        primary = min(overlapping, key=_desk_number)
        if len(overlapping) > 1:
            self._dirty = True
        for desk in overlapping:
            merged_box = _union_int(merged_box, desk.box)
            if desk.absence_alerted:
                events.append(
                    DeskMemoryEvent(
                        desk.name,
                        "ABSENT END",
                        max(0.0, timestamp - desk.last_occupied_at),
                    )
                )
            if desk is not primary:
                del self._desks[desk.name]
                self._dirty = True
        if primary.box != merged_box:
            self._dirty = True
        primary.box = merged_box
        primary.last_occupied_at = timestamp
        primary.absence_alerted = False
        primary.urgent_alerted = False
        return events, True

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("desks"), list):
                raise ValueError("document must contain a desks list")
            desks = {}
            for item in data["desks"]:
                if not isinstance(item, dict):
                    raise ValueError("desk entries must be mappings")
                name = item.get("id")
                raw_box = item.get("box")
                if (
                    not isinstance(name, str)
                    or not name.startswith("Desk ")
                    or not isinstance(raw_box, list)
                    or len(raw_box) != 4
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in raw_box
                    )
                ):
                    raise ValueError("desk entries require an ID and integer box")
                try:
                    number = int(name[5:])
                except ValueError as exc:
                    raise ValueError(f"{name!r} is not a valid desk ID") from exc
                if number < 1 or name != f"Desk {number}":
                    raise ValueError(f"{name!r} is not a valid desk ID")
                box = tuple(raw_box)
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"{name} has invalid bounds")
                if box[0] < 0 or box[1] < 0 or name in desks:
                    raise ValueError(f"{name} has invalid or duplicate bounds")
                created_at = _finite_number(item.get("created_at"), "created_at")
                last_occupied_at = _finite_number(
                    item.get("last_occupied_at"), "last_occupied_at"
                )
                absence_alerted = item.get("absence_alerted", False)
                urgent_alerted = item.get("urgent_alerted", False)
                if not isinstance(absence_alerted, bool) or not isinstance(
                    urgent_alerted, bool
                ):
                    raise ValueError(f"{name} alert state must be boolean")
                desks[name] = _RememberedDesk(
                    name,
                    box,
                    created_at,
                    last_occupied_at,
                    absence_alerted,
                    urgent_alerted,
                )
            maximum_id = max((_desk_number(desk) for desk in desks.values()), default=0)
            next_id = data.get("next_id", maximum_id + 1)
            if isinstance(next_id, bool) or not isinstance(next_id, int) or next_id < 1:
                raise ValueError("next_id must be a positive integer")
            if next_id <= maximum_id:
                raise ValueError("next_id must exceed all remembered desk IDs")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not load remembered desks from {self.path}: {exc}"
            ) from exc
        self._desks = desks
        self._next_id = next_id

    def _save(self, timestamp: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "camera": self.path.stem,
            "next_id": self._next_id,
            "desks": [
                {
                    "id": desk.name,
                    "box": list(desk.box),
                    "created_at": desk.created_at,
                    "last_occupied_at": desk.last_occupied_at,
                    "absence_alerted": desk.absence_alerted,
                    "urgent_alerted": desk.urgent_alerted,
                }
                for desk in sorted(self._desks.values(), key=_desk_number)
            ],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        self._last_saved_at = timestamp
        self._dirty = False


def _desk_number(desk: _RememberedDesk) -> int:
    return int(desk.name[5:])


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _center(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _normalized_distance(
    box: Sequence[float], center: Sequence[float], width: float, height: float
) -> float:
    box_center = _center(box)
    return math.hypot(
        (box_center[0] - center[0]) / width,
        (box_center[1] - center[1]) / height,
    )


def _union_float(
    first: Sequence[float], second: Sequence[float]
) -> Tuple[float, float, float, float]:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _union_int(first: Box, second: Box) -> Box:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _person_inside(person: Sequence[float], desk: Box) -> bool:
    center_x, center_y = _center(person)
    if desk[0] <= center_x <= desk[2] and desk[1] <= center_y <= desk[3]:
        return True
    area = max(1.0, (person[2] - person[0]) * (person[3] - person[1]))
    overlap = max(0.0, min(person[2], desk[2]) - max(person[0], desk[0])) * max(
        0.0, min(person[3], desk[3]) - max(person[1], desk[1])
    )
    return overlap / area > 0.15


def send_urgent_alert(
    camera: str,
    desk: str,
    absent_seconds: float,
    webhook_url: Optional[str] = None,
) -> None:
    LOGGER.error(
        "URGENT: %s ABSENT on camera %s for %.0f seconds",
        desk,
        camera,
        absent_seconds,
    )
    try:
        import winsound
    except ImportError:
        LOGGER.warning("Audio alert is unavailable on this platform.")
    else:
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

    if webhook_url:
        payload = json.dumps(
            {
                "event": "desk_urgent_absent",
                "camera": camera,
                "desk": desk,
                "absent_seconds": absent_seconds,
            }
        ).encode("utf-8")
        request = Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        thread = threading.Thread(
            target=_post_webhook,
            args=(request, camera, desk),
            name=f"desk-alert-{camera}-{desk}",
            daemon=True,
        )
        thread.start()


def _post_webhook(request: Request, camera: str, desk: str) -> None:
    try:
        with urlopen(request, timeout=3.0) as response:
            if response.status >= 400:
                LOGGER.warning(
                    "Urgent desk webhook for %s/%s returned HTTP %d",
                    camera,
                    desk,
                    response.status,
                )
    except (OSError, TimeoutError, URLError) as exc:
        LOGGER.warning("Urgent desk webhook for %s/%s failed: %s", camera, desk, exc)
