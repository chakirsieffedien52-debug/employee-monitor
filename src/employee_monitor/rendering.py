"""OpenCV status overlays and camera-grid composition."""

from collections.abc import Mapping, Sequence
from typing import Optional, Tuple

import cv2
import numpy as np

from employee_monitor.desk_memory import AbsentDesk
from employee_monitor.detection import Detection
from employee_monitor.status import ABSENT, PHONE_USAGE, WORKING
from employee_monitor.zones import DeskZone

COLORS = {
    WORKING: (45, 190, 45),
    ABSENT: (40, 40, 230),
    PHONE_USAGE: (35, 35, 230),
}
PHONE_COLOR = (0, 255, 255)


def draw_overlay(
    frame: np.ndarray,
    zones: Sequence[DeskZone],
    statuses: Mapping[str, str],
    people: Sequence[Detection],
    phones: Sequence[Detection],
    people_by_zone: Mapping[str, Tuple[Detection, ...]],
    fps: float,
) -> np.ndarray:
    output = frame.copy()
    person_zones = {
        id(person): zone.name
        for zone in zones
        for person in people_by_zone.get(zone.name, ())
    }
    for zone in zones:
        status = statuses.get(zone.name, ABSENT)
        color = COLORS[status]
        cv2.rectangle(output, (zone.x1, zone.y1), (zone.x2, zone.y2), color, 2)
        cv2.putText(
            output,
            f"{zone.name} - {status}",
            (zone.x1, max(20, zone.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    for person in people:
        desk = person_zones.get(id(person))
        color = COLORS.get(statuses.get(desk, WORKING), COLORS[WORKING])
        x1, y1, x2, y2 = (int(round(value)) for value in person.box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        identity = f"ID {person.track_id}" if person.track_id is not None else "Person"
        label = f"{identity} - {statuses.get(desk, WORKING) if desk else WORKING}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    for phone in phones:
        x1, y1, x2, y2 = (int(round(value)) for value in phone.box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            output,
            "Phone",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    working = sum(value == WORKING for value in statuses.values())
    alerts = sum(value in (ABSENT, PHONE_USAGE) for value in statuses.values())
    cv2.rectangle(output, (5, 5), (min(output.shape[1] - 5, 330), 37), (25, 25, 25), -1)
    cv2.putText(
        output,
        f"Working: {working} | Alerts: {alerts} | {fps:.1f} FPS",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def draw_detection_overlay(
    frame: np.ndarray,
    people: Sequence[Detection],
    phones: Sequence[Detection],
    statuses: Mapping[int, str],
    fps: float,
    absent_desks: Sequence[AbsentDesk] = (),
    urgent_message: Optional[str] = None,
    blink: bool = True,
) -> np.ndarray:
    output = frame.copy()
    for desk in absent_desks:
        x1, y1, x2, y2 = desk.box
        thickness = 6 if desk.urgent and blink else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        minutes = int(desk.absent_seconds // 60)
        seconds = int(desk.absent_seconds % 60)
        label = f"{desk.name} - ABSENT {minutes:02d}:{seconds:02d}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    for person in people:
        track_id = person.track_id
        status = statuses.get(track_id, WORKING) if track_id is not None else WORKING
        color = COLORS[status]
        x1, y1, x2, y2 = (int(round(value)) for value in person.box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        identity = f"ID {track_id}" if track_id is not None else "Person"
        cv2.putText(
            output,
            f"{identity} - {status}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    for phone in phones:
        x1, y1, x2, y2 = (int(round(value)) for value in phone.box)
        cv2.rectangle(output, (x1, y1), (x2, y2), PHONE_COLOR, 2)
        cv2.putText(
            output,
            "phone",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            PHONE_COLOR,
            1,
            cv2.LINE_AA,
        )

    phone_usage = sum(status == PHONE_USAGE for status in statuses.values())
    panel_text = (
        f"Persons: {len(people)} | Phone usage: {phone_usage} | "
        f"Absent desks: {len(absent_desks)} | FPS: {fps:.1f}"
    )
    (text_width, _), _ = cv2.getTextSize(panel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    cv2.rectangle(
        output,
        (5, 5),
        (min(output.shape[1] - 1, text_width + 24), 37),
        (25, 25, 25),
        -1,
    )
    cv2.putText(
        output,
        panel_text,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if urgent_message:
        banner_bottom = min(output.shape[0] - 1, 76)
        cv2.rectangle(
            output,
            (0, 42),
            (output.shape[1] - 1, banner_bottom),
            (0, 0, 150),
            -1,
        )
        cv2.putText(
            output,
            urgent_message,
            (12, min(output.shape[0] - 4, 66)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def compose_grid(frames: Mapping[str, np.ndarray], cell_size=(640, 360)) -> np.ndarray:
    if not frames:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    columns = max(1, int(np.ceil(np.sqrt(len(frames)))))
    rows = int(np.ceil(len(frames) / columns))
    grid = np.zeros((rows * cell_size[1], columns * cell_size[0], 3), dtype=np.uint8)
    for index, (name, frame) in enumerate(frames.items()):
        x, y = (index % columns) * cell_size[0], (index // columns) * cell_size[1]
        resized = cv2.resize(frame, cell_size)
        cv2.putText(
            resized,
            name,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        grid[y : y + cell_size[1], x : x + cell_size[0]] = resized
    return grid
