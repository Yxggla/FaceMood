# FaceMood Model Results

## Current Improved Model

- Architecture: deeper 4-block CNN with batch normalization, dropout, and adaptive average pooling.
- Training: 30 epochs on FER2013 7 classes.
- Optimization: Adam, learning-rate decay, weight decay, class-weighted cross entropy.
- Data augmentation: random horizontal flip, affine transform, and random erasing.
- Best validation accuracy: 60.96% at epoch 27.
- Test accuracy: 63.30%.

## Per-Class Test Summary

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| angry | 0.582 | 0.558 | 0.570 | 491 |
| disgust | 0.393 | 0.836 | 0.535 | 55 |
| fear | 0.464 | 0.246 | 0.322 | 528 |
| happy | 0.872 | 0.844 | 0.858 | 879 |
| neutral | 0.599 | 0.655 | 0.626 | 626 |
| sad | 0.475 | 0.534 | 0.502 | 594 |
| surprise | 0.681 | 0.849 | 0.756 | 416 |

## Comparison With First Baseline

- First baseline test accuracy: 60.80%.
- Improved model test accuracy: 63.30%.
- Absolute improvement: +2.50 percentage points.
- The biggest practical improvement is the `disgust` recall, which rose from 0.164 to 0.836 after class weighting and augmentation.
- `fear` remains the weakest class and should be discussed as a limitation.

## Files Generated Locally

These files are intentionally not pushed to GitHub:

- `models/exported/emotion_cnn.pt`
- `results/metrics/training_history.json`
- `results/metrics/test_classification_report.json`
- `results/figures/training_history.png`
- `results/figures/test_confusion_matrix.png`

