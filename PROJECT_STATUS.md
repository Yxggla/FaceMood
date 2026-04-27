# FaceMood Project Status

## Done

- Public GitHub repository.
- Cross-platform launch scripts for macOS and Windows.
- Tkinter project launcher that runs without the emotion model.
- Dataset counting for FER2013 7 classes.
- Real-time pipeline code structure: camera, face detection, landmarks, alignment, prediction, visualization.
- Training and evaluation script structure.
- Report and presentation outlines based on course requirements.

## Not Done Yet

- Dependency installation on each teammate's machine.
- Camera demo validation after installing OpenCV and MediaPipe.
- Emotion model training.
- Evaluation metrics and confusion matrix.
- Screenshots and demo video for report/PPT.
- Final report text and final slide deck.

## Recommended Next Work

1. Each teammate runs `python run.py` and confirms the launcher opens.
2. One teammate installs dependencies and validates the camera demo.
3. Model owner trains the CNN and saves `models/exported/emotion_cnn.pt`.
4. Evaluation owner runs `train/evaluate.py` and saves metrics/figures.
5. Report/PPT owner fills in results using files in `results/`.

