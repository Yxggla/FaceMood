from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "exported" / "emotion_cnn_v2_kaggle.pt"
MAIN = ROOT / "src" / "main.py"


def _check_runtime() -> None:
    missing: list[str] = []
    for module in ("numpy", "cv2", "torch"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing Python packages for cnn_v2 demo: {joined}. "
            "Activate the project venv and run `pip install -r requirements.txt`."
        )
    if not MODEL.exists():
        raise SystemExit(f"Model file not found: {MODEL}")


if __name__ == "__main__":
    _check_runtime()
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(MAIN),
                "--model",
                str(MODEL),
                "--no-landmarks",
                "allow",
                "--use-geo-rules",
                "off",
                "--use-emotion-bias",
                "off",
                "--min-emotion-conf",
                "0.0",
                "--switch-margin",
                "0.02",
            ]
        )
    )
