"""Frame-based desk status smoothing."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from employee_monitor.detection import Detection

WORKING = "WORKING"
ABSENT = "ABSENT"
PHONE_USAGE = "PHONE USAGE"
STATUSES = (WORKING, ABSENT, PHONE_USAGE)


@dataclass
class _DeskState:
    status: str = ABSENT
    candidate: Optional[str] = None
    candidate_frames: int = 0
    durations: Dict[str, float] = field(
        default_factory=lambda: {status: 0.0 for status in STATUSES}
    )


@dataclass(frozen=True)
class StatusChange:
    desk: str
    status: str


class StatusTracker:
    def __init__(self, desks: Iterable[str], stable_frames: int, fps: float) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1.")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive.")
        self.stable_frames = stable_frames
        self.fps = fps
        self._states = {desk: _DeskState() for desk in desks}
        if not self._states:
            raise ValueError("At least one desk is required.")

    def add_desks(self, desks: Iterable[str]) -> None:
        for desk in desks:
            if desk not in self._states:
                self._states[desk] = _DeskState()

    @property
    def statuses(self) -> Dict[str, str]:
        return {desk: state.status for desk, state in self._states.items()}

    def update(
        self, observations: Mapping[str, str], elapsed_frames: int = 1
    ) -> List[StatusChange]:
        if elapsed_frames < 1:
            raise ValueError("elapsed_frames must be at least 1.")
        elapsed_seconds = elapsed_frames / self.fps
        changes: List[StatusChange] = []
        for desk, state in self._states.items():
            observed = observations.get(desk, ABSENT)
            if observed not in STATUSES:
                raise ValueError(f"Unknown status {observed!r} for desk {desk!r}.")
            state.durations[state.status] += elapsed_seconds
            if observed == state.status:
                state.candidate = None
                state.candidate_frames = 0
                continue
            if observed != state.candidate:
                state.candidate = observed
                state.candidate_frames = 1
            else:
                state.candidate_frames += 1
            if state.candidate_frames >= self.stable_frames:
                state.status = observed
                state.candidate = None
                state.candidate_frames = 0
                changes.append(StatusChange(desk, state.status))
        return changes

    def advance(self, elapsed_frames: int = 1) -> None:
        if elapsed_frames < 1:
            raise ValueError("elapsed_frames must be at least 1.")
        elapsed_seconds = elapsed_frames / self.fps
        for state in self._states.values():
            state.durations[state.status] += elapsed_seconds

    def summary(self) -> List[Tuple[str, str, float]]:
        return [
            (desk, status, state.durations[status])
            for desk, state in self._states.items()
            for status in STATUSES
        ]


@dataclass
class _PersonState:
    status: Optional[str] = None
    candidate: Optional[str] = None
    candidate_frames: int = 0


class PersonStatusTracker:
    def __init__(self, stable_frames: int) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1.")
        self.stable_frames = stable_frames
        self._states: Dict[int, _PersonState] = {}

    @property
    def statuses(self) -> Dict[int, str]:
        return {
            track_id: state.status or WORKING
            for track_id, state in self._states.items()
        }

    def update(
        self,
        observations: Mapping[int, str],
        visible_ids: Iterable[int],
    ) -> List[StatusChange]:
        visible = set(visible_ids)
        self._states = {
            track_id: state
            for track_id, state in self._states.items()
            if track_id in visible
        }
        changes = []
        for track_id in sorted(visible):
            state = self._states.setdefault(track_id, _PersonState())
            observed = observations.get(track_id)
            if observed is None:
                continue
            if observed not in (WORKING, PHONE_USAGE):
                raise ValueError(
                    f"Unknown status {observed!r} for person ID {track_id}."
                )
            if observed == state.status:
                state.candidate = None
                state.candidate_frames = 0
                continue
            if observed != state.candidate:
                state.candidate = observed
                state.candidate_frames = 1
            else:
                state.candidate_frames += 1
            if state.candidate_frames >= self.stable_frames:
                state.status = observed
                state.candidate = None
                state.candidate_frames = 0
                changes.append(StatusChange(f"ID {track_id}", observed))
        return changes


def assign_person_phone_statuses(
    people: Sequence[Detection],
    phones: Sequence[Detection],
    proximity_ratio: float,
) -> Dict[int, str]:
    assignments: Dict[int, List[Detection]] = {
        person.track_id: []
        for person in people
        if person.track_id is not None and person.observed
    }
    tracked_people = [
        person for person in people if person.track_id is not None and person.observed
    ]
    for phone in phones:
        if not tracked_people:
            break
        phone_x = (phone.box[0] + phone.box[2]) / 2.0
        phone_y = (phone.box[1] + phone.box[3]) / 2.0
        person = min(
            tracked_people,
            key=lambda item: _upper_body_distance(item, phone_x, phone_y),
        )
        assignments[person.track_id].append(phone)

    statuses = {}
    for person in tracked_people:
        height = max(1.0, person.box[3] - person.box[1])
        width = max(1.0, person.box[2] - person.box[0])
        upper_body = (
            person.box[0] - width * proximity_ratio,
            person.box[1] - height * proximity_ratio * 0.5,
            person.box[2] + width * proximity_ratio,
            person.box[1] + height * (0.65 + proximity_ratio * 0.5),
        )
        phone_use = False
        for phone in assignments[person.track_id]:
            center_x = (phone.box[0] + phone.box[2]) / 2.0
            center_y = (phone.box[1] + phone.box[3]) / 2.0
            overlaps_upper_body = (
                phone.box[0] <= upper_body[2]
                and phone.box[2] >= upper_body[0]
                and phone.box[1] <= upper_body[3]
                and phone.box[3] >= upper_body[1]
            )
            near_wrist = False
            if person.keypoints:
                wrist_distance = max(24.0, height * 0.15)
                near_wrist = any(
                    score >= 0.2
                    and math.hypot(x - center_x, y - center_y) < wrist_distance
                    for index in (9, 10)
                    if index < len(person.keypoints)
                    for x, y, score in (person.keypoints[index],)
                )
            if overlaps_upper_body or near_wrist:
                phone_use = True
                break
        statuses[person.track_id] = PHONE_USAGE if phone_use else WORKING
    return statuses


def _upper_body_distance(person: Detection, x: float, y: float) -> float:
    width = max(1.0, person.box[2] - person.box[0])
    height = max(1.0, person.box[3] - person.box[1])
    center_x = (person.box[0] + person.box[2]) / 2.0
    center_y = person.box[1] + height * 0.4
    return math.hypot((x - center_x) / width, (y - center_y) / height)
