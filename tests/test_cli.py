from types import SimpleNamespace

import numpy as np

from employee_monitor.capture import FramePacket
from employee_monitor.cli import _desk_memory_timestamp


def test_file_desk_memory_timestamp_uses_frame_index_and_fps(monkeypatch):
    monkeypatch.setattr("employee_monitor.cli.time.time", lambda: 1000.0)
    runtime = SimpleNamespace(
        worker=SimpleNamespace(is_live=False),
        memory_time_origin=None,
    )

    start = _desk_memory_timestamp(runtime, FramePacket(np.zeros((1, 1)), 100, 0, 25))
    later = _desk_memory_timestamp(runtime, FramePacket(np.zeros((1, 1)), 150, 0, 25))

    assert later - start == 2.0


def test_live_desk_memory_timestamp_uses_wall_clock(monkeypatch):
    monkeypatch.setattr("employee_monitor.cli.time.time", lambda: 1234.5)
    runtime = SimpleNamespace(worker=SimpleNamespace(is_live=True))

    timestamp = _desk_memory_timestamp(
        runtime, FramePacket(np.zeros((1, 1)), 150, 0, 25)
    )

    assert timestamp == 1234.5
