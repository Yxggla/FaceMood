from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from facemood.auto_capture import AutoEmotionScreenshotter
from facemood.camera import Camera
from facemood.config import (
    BBOX_EMA_ALPHA,
    CAMERA_INDEX,
    EMOTION_EMA_ALPHA,
    EMOTION_SWITCH_MARGIN,
    EMOTION_SWITCH_HOLD_FRAMES,
    FACE_DETECTOR,
    FACE_LOST_TOLERANCE,
    FACE_MIN_DETECTION_CONFIDENCE,
    FACE_MIN_SIZE,
    MIN_EMOTION_CONF,
    NO_LANDMARKS_POLICY,
    MODEL_PATH,
    PROJECT_ROOT,
    STABILIZE_SINGLE_FACE,
)
from facemood.predictor import FaceMoodPredictor
from facemood.visualizer import draw_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FaceMood real-time demo.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu, mps, cuda")
    parser.add_argument("--detector", choices=["mediapipe", "haar", "auto"], default=FACE_DETECTOR)
    parser.add_argument("--min-det-conf", type=float, default=FACE_MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--min-face-size", type=int, default=FACE_MIN_SIZE)
    parser.add_argument("--no-landmarks", choices=["discard", "unknown", "allow", "infer"], default=NO_LANDMARKS_POLICY)
    parser.add_argument("--stable", choices=["on", "off"], default="on" if STABILIZE_SINGLE_FACE else "off")
    parser.add_argument("--use-geo-rules", choices=["on", "off"], default="on")
    parser.add_argument("--use-emotion-bias", choices=["on", "off"], default="on")
    parser.add_argument("--bbox-alpha", type=float, default=BBOX_EMA_ALPHA)
    parser.add_argument("--emotion-alpha", type=float, default=EMOTION_EMA_ALPHA)
    parser.add_argument("--switch-margin", type=float, default=EMOTION_SWITCH_MARGIN)
    parser.add_argument("--switch-hold-frames", type=int, default=EMOTION_SWITCH_HOLD_FRAMES)
    parser.add_argument("--face-lost-tolerance", type=int, default=FACE_LOST_TOLERANCE)
    parser.add_argument("--min-emotion-conf", type=float, default=MIN_EMOTION_CONF)
    parser.add_argument("--auto-screenshots", choices=["on", "off"], default="on")
    parser.add_argument("--auto-shots-per-emotion", type=int, default=10)
    parser.add_argument("--auto-stable-frames", type=int, default=5)
    parser.add_argument("--auto-interval-frames", type=int, default=3)
    parser.add_argument("--auto-min-conf", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FaceMoodPredictor(
        model_path=args.model,
        device=args.device,
        stable=args.stable == "on",
        detector=args.detector,
        min_detection_confidence=args.min_det_conf,
        min_face_size=args.min_face_size,
        no_landmarks_policy=args.no_landmarks,
        bbox_alpha=args.bbox_alpha,
        emotion_alpha=args.emotion_alpha,
        switch_margin=args.switch_margin,
        switch_hold_frames=args.switch_hold_frames,
        face_lost_tolerance=args.face_lost_tolerance,
        min_emotion_conf=args.min_emotion_conf,
        use_geo_rules=args.use_geo_rules == "on",
        use_emotion_bias=args.use_emotion_bias == "on",
    )
    screenshots_dir = PROJECT_ROOT / "results" / "screenshots"
    videos_dir = PROJECT_ROOT / "results" / "videos"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    auto_screenshotter = (
        AutoEmotionScreenshotter(
            screenshots_dir / "auto",
            shots_per_emotion=args.auto_shots_per_emotion,
            stable_frames=args.auto_stable_frames,
            interval_frames=args.auto_interval_frames,
            min_confidence=args.auto_min_conf,
        )
        if args.auto_screenshots == "on"
        else None
    )
    if auto_screenshotter is not None:
        print(
            "Auto screenshots enabled: "
            f"{auto_screenshotter.shots_per_emotion} per emotion after "
            f"{auto_screenshotter.stable_frames} stable frames. "
            f"Output: {auto_screenshotter.output_dir}"
        )
    recorder = DemoRecorder(videos_dir)
    fps_meter = FpsMeter()

    with Camera(index=args.camera) as camera:
        while True:
            frame = camera.read()
            frame = cv2.flip(frame, 1)  # 水平镜像翻转，营造自拍镜效果
            predictions = predictor.predict_frame(frame)
            fps = fps_meter.tick()
            auto_status = _auto_status(auto_screenshotter)
            output = draw_predictions(
                frame,
                predictions,
                fps=fps,
                recording=recorder.is_recording,
                auto_status=auto_status,
            )
            if auto_screenshotter is not None:
                for result in auto_screenshotter.update(output, predictions):
                    print(
                        f"Auto screenshot [{result.emotion} "
                        f"{result.count}/{auto_screenshotter.shots_per_emotion}]: {result.path}"
                    )
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


def _auto_status(auto_screenshotter: AutoEmotionScreenshotter | None) -> str | None:
    if auto_screenshotter is None:
        return None
    if auto_screenshotter.is_complete:
        return f"AUTO done {auto_screenshotter.total_saved}/{auto_screenshotter.total_target}"
    return f"AUTO {auto_screenshotter.total_saved}/{auto_screenshotter.total_target}"


if __name__ == "__main__":
    main()
