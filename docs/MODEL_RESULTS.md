# FaceMood Model Results

## Current Model Summary

The project now keeps three tracked model lines:

- Baseline CNN: archived comparison model.
- `cnn_v2`: stronger custom grayscale CNN used as a comparison runtime.
- `resnet18`: current recommended demo/runtime model.

## Current Best Model

- Architecture: `resnet18`
- Training: 30 epochs on FER2013 7 classes
- Input: 224x224, grayscale expanded to 3 channels with ImageNet normalization
- Test accuracy: 69.30%
- Macro F1: 0.6879

## Secondary Comparison Model

- Architecture: `cnn_v2`
- Training: 40 epochs on FER2013 7 classes
- Input: 48x48 grayscale
- Test accuracy: 66.09%
- Macro F1: 0.6246

## Archived Baseline

- Architecture: deeper 4-block CNN with batch normalization, dropout, and adaptive average pooling
- Training: 30 epochs on FER2013 7 classes
- Best validation accuracy: 60.96% at epoch 27
- Test accuracy: 63.30%

## Accuracy Comparison

| Model | Test Accuracy | Notes |
|---|---:|---|
| Baseline CNN | 63.30% | Archived local baseline |
| `cnn_v2` | 66.09% | Stronger custom grayscale CNN |
| `resnet18` | 69.30% | Current recommended runtime model |

## Stored Local Model Files

- `models/exported/emotion_cnn.pt`
- `models/exported/emotion_cnn_v2_kaggle.pt`
- `models/exported/emotion_resnet18_kaggle.pt`
- `results/metrics/cnn_v2_kaggle_summary.json`
- `results/metrics/resnet18_kaggle_summary.json`

