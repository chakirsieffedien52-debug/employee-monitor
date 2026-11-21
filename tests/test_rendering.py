import numpy as np

from employee_monitor.desk_memory import AbsentDesk
from employee_monitor.rendering import draw_detection_overlay


def test_detection_mode_draws_no_fixed_boxes_when_nothing_is_detected():
    frame = np.zeros((100, 800, 3), dtype=np.uint8)

    output = draw_detection_overlay(frame, [], [], {}, fps=0.0)

    assert np.array_equal(output[40:, :], frame[40:, :])
    assert np.array_equal(output[:, 600:, :], frame[:, 600:, :])


def test_detection_mode_draws_an_absent_desk_and_urgent_banner():
    frame = np.zeros((160, 800, 3), dtype=np.uint8)
    desk = AbsentDesk("Desk 1", (80, 100, 140, 145), 42.0, True)

    output = draw_detection_overlay(
        frame,
        [],
        [],
        {},
        fps=1.0,
        absent_desks=[desk],
        urgent_message="URGENT: Desk 1 ABSENT",
        blink=True,
    )

    assert tuple(output[100, 80]) == (0, 0, 255)
    assert np.any(output[42:77, :, 2] > 0)
