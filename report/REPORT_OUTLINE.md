# FaceMood Report Outline

Target length: 3 pages, maximum 4 pages, PDF submission.

## 1. Objectives

FaceMood aims to build a real-time facial expression recognition system using Python. The system detects faces from a webcam stream, localizes facial landmarks, predicts FER2013 7-class emotions, and visualizes emotion labels, confidence scores, and current emotion distribution.

## 2. Proposed Methods

- Input: webcam frames or local test images/videos.
- Face detection: OpenCV Haar Cascade baseline, replaceable by MTCNN/RetinaFace.
- Landmark localization: MediaPipe Face Mesh for eyes, nose, and mouth points.
- Face preprocessing: crop and align face region, convert to grayscale 48x48 image.
- Emotion recognition: lightweight CNN trained on FER2013 7 classes.
- Visualization: bounding boxes, landmarks, labels, confidence values, and emotion distribution.

## 3. Findings and Results

To be filled after experiments:

- dataset class distribution from `report/DATASET_SUMMARY.md`
- best validation accuracy: 60.96%
- test accuracy: 63.30%
- confusion matrix from `results/figures/test_confusion_matrix.png`
- training curve from `results/figures/training_history.png`
- single-person webcam test
- multi-person webcam test
- lighting condition test
- screenshots and short demo video evidence

## 4. Conclusion and Discussion

Discuss what worked, limitations such as remaining confusion in `fear`, webcam lighting sensitivity, and possible improvements such as stronger detectors or a pretrained backbone. Mention that class weighting improved the minority `disgust` recall but did not fully solve all class confusion.

## 5. Individual Contributions

| Member | Contribution |
|---|---|
| Member 1 | Project integration, repository, real-time pipeline |
| Member 2 | Face detection |
| Member 3 | Landmarks and alignment |
| Member 4 | Emotion model training |
| Member 5 | Visualization and effects |
| Member 6 | Evaluation, report, slides |
