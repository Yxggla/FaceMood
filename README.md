# FaceMood

FaceMood is a Python real-time facial expression recognition demo for the COMM7350 course project. It uses FER2013's 7 emotion classes:

```python
["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
```

## Project Layout

```text
data/                  FER2013 CSV, image folders, and local samples
models/                Training checkpoints and exported demo weights
src/facemood/          Real-time camera, detection, prediction, visualization
train/                 Dataset preparation, training, and evaluation scripts
assets/stickers/       Optional transparent PNG sticker effects
results/               Figures, metrics, screenshots, and demo videos
report/                Course report materials
slides/                Course presentation materials
tests/                 Lightweight project tests
```

## Setup

Python 3.10 is recommended.

If you only want to open the project launcher first, no extra packages are required:

```bash
python3 run.py
```

On macOS, double-click `start_mac.command`. On Windows, double-click `start_windows.bat`.

For detailed teammate setup instructions, read `START_HERE.md`.

To run the full camera demo and training pipeline, install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

The project reads the existing image dataset at `data/fer2013_7cls_images`.
The FER2013 data folders are intentionally ignored by Git because they are large local assets.

```bash
python train/train_emotion.py --epochs 10
python train/evaluate.py --weights models/exported/emotion_cnn.pt
```

For a quick smoke test:

```bash
python train/train_emotion.py --epochs 1 --limit-train 256 --limit-val 128
```

## Run Demo

```bash
python src/main.py
```

If `models/exported/emotion_cnn.pt` does not exist yet, the demo still opens the camera and draws detections/landmarks, but emotion labels are shown as `unknown`.
