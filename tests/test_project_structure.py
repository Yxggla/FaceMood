from pathlib import Path


def test_expected_directories_exist():
    root = Path(__file__).resolve().parents[1]
    for path in [
        "data/fer2013",
        "data/fer2013_7cls_images/train",
        "models/checkpoints",
        "models/exported",
        "src/facemood",
        "train",
        "results/metrics",
        "report",
        "slides",
    ]:
        assert (root / path).exists()


def test_emotion_classes_are_fer2013_7_class_order():
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from facemood.config import EMOTION_CLASSES

    assert EMOTION_CLASSES == ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

