from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from facemood.camera import Camera
from facemood.config import CAMERA_INDEX, MODEL_PATH, PROJECT_ROOT
from facemood.predictor import FaceMoodPredictor
from facemood.visualizer import draw_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FaceMood real-time demo.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu, mps, cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FaceMoodPredictor(model_path=args.model, device=args.device)
    screenshots_dir = PROJECT_ROOT / "results" / "screenshots"
    videos_dir = PROJECT_ROOT / "results" / "videos"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    recorder = DemoRecorder(videos_dir)
    fps_meter = FpsMeter()

    with Camera(index=args.camera) as camera:
        while True:
            frame = camera.read()
            predictions = predictor.predict_frame(frame)
            fps = fps_meter.tick()
            output = draw_predictions(frame, predictions, fps=fps, recording=recorder.is_recording)
            recorder.write(output)
            cv2.imshow("FaceMood", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = screenshots_dir / f"facemood_{_timestamp()}.png"
                cv2.imwrite(str(path), output)
                print(f"Saved screenshot: {path}")
            if key == ord("r"):
                if recorder.is_recording:
                    path = recorder.stop()
                    print(f"Saved video: {path}")
                else:
                    recorder.start(output)
                    print("Recording started. Press r again to stop.")
    recorder.stop()


class FpsMeter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.last_time: float | None = None
        self.fps: float | None = None

    def tick(self) -> float | None:
        now = time.perf_counter()
        if self.last_time is None:
            self.last_time = now
            return self.fps
        elapsed = max(now - self.last_time, 1e-6)
        instant = 1.0 / elapsed
        self.fps = instant if self.fps is None else self.fps * self.smoothing + instant * (1 - self.smoothing)
        self.last_time = now
        return self.fps


class DemoRecorder:
    def __init__(self, output_dir: Path, fps: float = 20.0) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self.writer is not None

    def start(self, frame) -> None:
        self.path = self.output_dir / f"facemood_demo_{_timestamp()}.mp4"
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"Unable to start video recording at {self.path}")

    def write(self, frame) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def stop(self) -> Path | None:
        path = self.path
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.path = None
        return path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
