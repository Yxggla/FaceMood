from __future__ import annotations

import argparse

import cv2

from facemood.camera import Camera
from facemood.config import CAMERA_INDEX, MODEL_PATH
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
    with Camera(index=args.camera) as camera:
        while True:
            frame = camera.read()
            predictions = predictor.predict_frame(frame)
            output = draw_predictions(frame, predictions)
            cv2.imshow("FaceMood", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break


if __name__ == "__main__":
    main()

