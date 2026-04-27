from __future__ import annotations

import cv2


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera is not open")
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        return frame

