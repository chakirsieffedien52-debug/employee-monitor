"""Optional local face blurring for previews and recorded output."""

import logging
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class FaceBlur:
    def __init__(self, enabled: bool) -> None:
        self._classifier = None
        if enabled:
            cascade_path = (
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            )
            classifier = cv2.CascadeClassifier(str(cascade_path))
            if classifier.empty():
                raise RuntimeError("Could not load OpenCV's face detection cascade.")
            self._classifier = classifier

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self._classifier is None:
            return frame
        output = frame.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        faces = self._classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
        )
        for x, y, width, height in faces:
            region = output[y : y + height, x : x + width]
            if region.size:
                output[y : y + height, x : x + width] = cv2.GaussianBlur(
                    region, (31, 31), 0
                )
        return output
