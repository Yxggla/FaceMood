# FaceMood Project Status

## Done

- Public GitHub repository.
- Python runtime entrypoints for `cnn_v2` and `resnet18`.
- Tkinter project launcher that runs without the emotion model.
- Dataset counting for FER2013 7 classes.
- Real-time pipeline code structure: camera, face detection, landmarks, alignment, prediction, visualization.
- Training and evaluation script structure.
- Report and presentation outlines based on course requirements.
- Baseline CNN result retained for archive and comparison.
- `cnn_v2` result retained locally with 66.09% test accuracy.
- `resnet18` result retained locally with 69.30% test accuracy.
- Demo controls for screenshot and video recording.

## Not Done Yet

- Dependency installation on each teammate's machine.
- Camera demo validation after installing OpenCV and MediaPipe.
- Final chosen demo screenshots and demo video for report/PPT.
- Final report text and final slide deck.

## Recommended Next Work

1. Each teammate runs `python run.py` and confirms the launcher opens.
2. One teammate installs dependencies and validates both `python3 run_demo_resnet18.py` and `python3 run_demo_cnn_v2.py`.
3. Model owner keeps `models/exported/emotion_resnet18_kaggle.pt` as the primary demo/runtime model.
4. Evaluation owner uses metrics/figures from `results/`.
5. Report/PPT owner fills in final slides and report using the latest tracked results.
