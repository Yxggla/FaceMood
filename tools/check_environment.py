from __future__ import annotations

import importlib.util
import platform
import sys

OPTIONAL_DEPENDENCIES = {
    "opencv-python": "cv2",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "mediapipe": "mediapipe",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
}


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print()
    for package, module in OPTIONAL_DEPENDENCIES.items():
        status = "installed" if importlib.util.find_spec(module) else "missing"
        print(f"{package:15} {status}")
    print()
    print("Install missing packages with: pip install -r requirements.txt")


if __name__ == "__main__":
    main()

