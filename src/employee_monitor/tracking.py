"""Independent per-camera person tracking and short-term box persistence."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

from employee_monitor.detection import PERSON_CLASS_ID, Detection
from employee_monitor.zones import intersection_over_union


@dataclass
class _PersonTrack:
    detection: Detection
    missed_frames: int = 0


class PersonTracker:
    def __init__(
        self, persistence_frames: int = 3, iou_threshold: float = 0.15
    ) -> None:
        if persistence_frames < 0:
            raise ValueError("persistence_frames cannot be negative.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")
        self.persistence_frames = persistence_frames
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, _PersonTrack] = {}
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> Tuple[Detection, ...]:
        people = [item for item in detections if item.class_id == PERSON_CLASS_ID]
        other = [item for item in detections if item.class_id != PERSON_CLASS_ID]
        candidates = []
        for track_id, track in self._tracks.items():
            for index, person in enumerate(people):
                overlap = intersection_over_union(track.detection.box, person.box)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, track_id, index))
        candidates.sort(reverse=True)
        matched_tracks = set()
        matched_people = set()
        visible: List[Detection] = []
        for _, track_id, index in candidates:
            if track_id in matched_tracks or index in matched_people:
                continue
            detection = replace(people[index], track_id=track_id, observed=True)
            self._tracks[track_id] = _PersonTrack(detection)
            visible.append(detection)
            matched_tracks.add(track_id)
            matched_people.add(index)

        for index, person in enumerate(people):
            if index in matched_people:
                continue
            track_id = self._next_id
            self._next_id += 1
            detection = replace(person, track_id=track_id, observed=True)
            self._tracks[track_id] = _PersonTrack(detection)
            visible.append(detection)
            matched_tracks.add(track_id)

        for track_id in list(self._tracks):
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.persistence_frames:
                del self._tracks[track_id]
            else:
                visible.append(replace(track.detection, observed=False))

        return tuple(visible + other)
