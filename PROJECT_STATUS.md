# FaceMood Project Status

## Done

- Public GitHub repository.
- Cross-platform launch scripts for macOS and Windows.
- Tkinter project launcher that runs without the emotion model.
- Dataset counting for FER2013 7 classes.
- Real-time pipeline code structure: camera, face detection, landmarks, alignment, prediction, visualization.
- Training and evaluation script structure.
- Report and presentation outlines based on course requirements.
- Improved 30-epoch model trained locally with around 63.3% test accuracy.
- Demo controls for screenshot and video recording.

## Not Done Yet

- Dependency installation on each teammate's machine.
- Camera demo validation after installing OpenCV and MediaPipe.
- Final chosen demo screenshots and demo video for report/PPT.
- Final report text and final slide deck.

## Recommended Next Work

1. Each teammate runs `python run.py` and confirms the launcher opens.
2. One teammate installs dependencies and validates the camera demo.
3. Model owner keeps `models/exported/emotion_cnn.pt` for local demo or submission packaging.
4. Evaluation owner uses metrics/figures from `results/`.
5. Report/PPT owner fills in final slides and report using the improved model result.
