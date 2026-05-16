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

For detailed teammate setup instructions, read `START_HERE.md`.

The launcher can show dataset counts, preview sample images, check dependencies, and generate a dataset report without running the emotion model.

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
python train/train_emotion.py --arch cnn_v2 --epochs 40 --class-weights
python train/evaluate.py --weights models/exported/emotion_resnet18_kaggle.pt
```

For a quick smoke test:

```bash
python train/train_emotion.py --arch cnn_v2 --epochs 1 --limit-train 256 --limit-val 128
```

## Dataset Report

This command does not train or run the model. It only summarizes the local FER2013 data:

```bash
python tools/generate_dataset_report.py
```

Outputs:

```text
results/metrics/dataset_summary.json
results/metrics/dataset_summary.csv
report/DATASET_SUMMARY.md
```

## Run Demo

```bash
python3 run_demo_resnet18.py
```

Alternative demo entry:

```bash
python3 run_demo_cnn_v2.py
```

These two scripts are the supported model-specific runtime entrypoints:

- `run_demo_resnet18.py`: current default demo path, loads `models/exported/emotion_resnet18_kaggle.pt`
- `run_demo_cnn_v2.py`: comparison demo path, loads `models/exported/emotion_cnn_v2_kaggle.pt`

The original baseline CNN checkpoint is kept in the repository as an archived training result, but it is no longer used as a demo runtime target.

If the selected model file does not exist yet, the demo still opens the camera and draws detections/landmarks, but emotion labels are shown as `unknown`.

Demo controls:

- `s`: save screenshot to `results/screenshots/`
- `r`: start/stop video recording to `results/videos/`
- `q` or `Esc`: quit

The demo overlays FPS, current emotion distribution, and a recording indicator.

## Optional Model Tuning

The training pipeline still supports the archived baseline CNN, `cnn_v2`, and `resnet18`.

Recommended training commands:

```bash
python train/train_emotion.py --arch cnn_v2 --epochs 40 --class-weights
python train/train_emotion.py --arch resnet18 --epochs 30 --batch-size 64 --class-weights
```

Current tracked results:

- Baseline CNN: `63.30%` test accuracy
- `cnn_v2`: `66.09%` test accuracy
- `resnet18`: `69.30%` test accuracy
- Current recommended demo/runtime model: `resnet18`
