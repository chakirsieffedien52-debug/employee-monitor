"""Camera capture workers with bounded latest-frame queues."""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

from employee_monitor.config import Source, source_is_network

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    index: int
    captured_at: float
    fps: float


class CameraWorker:
    def __init__(
        self,
        name: str,
        source: Source,
        reconnect_initial_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self.source = source
        self.reconnect_initial_seconds = reconnect_initial_seconds
        self.reconnect_max_seconds = reconnect_max_seconds
        self._frames: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_live = isinstance(source, int) or source_is_network(str(source))
        self._fps = 0.0
        self._resolution = (0, 0)
        self.frames_read = 0
        self.reconnects = 0

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    @property
    def is_live(self) -> bool:
        return self._is_live

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self):
        return self._resolution

    def start(self) -> "CameraWorker":
        if self._thread is not None:
            raise RuntimeError(f"Camera worker {self.name!r} has already been started.")
        self._thread = threading.Thread(
            target=self._run, name=f"camera-{self.name}", daemon=True
        )
        self._thread.start()
        return self

    def get_latest(self, timeout: Optional[float] = None) -> Optional[FramePacket]:
        try:
            packet = self._frames.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try:
                packet = self._frames.get_nowait()
            except queue.Empty:
                return packet

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        backoff = self.reconnect_initial_seconds
        try:
            while not self._stop.is_set():
                capture = cv2.VideoCapture(self.source)
                if not capture.isOpened():
                    capture.release()
                    if not self._is_live:
                        LOGGER.error(
                            "Could not open video source for camera %s", self.name
                        )
                        break
                    self.reconnects += 1
                    LOGGER.warning(
                        "Camera %s connection failed; retrying in %.1f seconds",
                        self.name,
                        backoff,
                    )
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2.0, self.reconnect_max_seconds)
                    continue
                backoff = self.reconnect_initial_seconds
                self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                self._resolution = (
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                index = 0
                playback_started = time.monotonic()
                try:
                    while not self._stop.is_set():
                        success, frame = capture.read()
                        if not success:
                            if self._is_live:
                                self.reconnects += 1
                                LOGGER.warning(
                                    "Camera %s stream ended; reconnecting", self.name
                                )
                                break
                            return
                        self.frames_read += 1
                        self._publish(
                            FramePacket(frame, index, time.monotonic(), self._fps)
                        )
                        index += 1
                        if not self._is_live and self._fps > 0:
                            target_time = playback_started + index / self._fps
                            self._stop.wait(max(0.0, target_time - time.monotonic()))
                finally:
                    capture.release()
                if self._is_live and not self._stop.is_set():
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2.0, self.reconnect_max_seconds)
        finally:
            self._finished.set()

    def _publish(self, packet: FramePacket) -> None:
        try:
            self._frames.put_nowait(packet)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(packet)
            except queue.Full:
                LOGGER.debug("Dropped frame for camera %s", self.name)


def test_camera(source: Union[str, int], timeout: float = 5.0) -> Optional[dict]:
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            return None
        deadline = time.monotonic() + timeout
        success, frame = False, None
        while time.monotonic() < deadline:
            success, frame = capture.read()
            if success:
                break
            time.sleep(0.05)
        if not success or frame is None:
            return None
        return {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        }
    finally:
        capture.release()
