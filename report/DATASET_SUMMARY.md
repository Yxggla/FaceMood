# FER2013 Dataset Summary

Generated at: `2026-04-27T14:19:11`

Data source: `/Users/dongbaba/Documents/FaceMood/data/fer2013_7cls_images`

## Class Counts

| Emotion | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| angry | 3995 | 467 | 491 | 4953 |
| disgust | 436 | 56 | 55 | 547 |
| fear | 4097 | 496 | 528 | 5121 |
| happy | 7215 | 895 | 879 | 8989 |
| neutral | 4965 | 607 | 626 | 6198 |
| sad | 4830 | 653 | 594 | 6077 |
| surprise | 3171 | 415 | 416 | 4002 |
| **TOTAL** | **28709** | **3589** | **3589** | **35887** |

## Notes for Report

- The project uses all 7 FER2013 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise.
- The `disgust` class is heavily underrepresented compared with the other classes.
- This imbalance should be mentioned in the findings/discussion section after model evaluation.
