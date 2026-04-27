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


def test_dataset_summary_payload_contains_totals():
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from facemood.reporting import build_dataset_summary

    payload = build_dataset_summary(root / "data" / "fer2013_7cls_images")
    assert payload["total_images"] == 35887
    assert payload["split_totals"] == {"train": 28709, "val": 3589, "test": 3589}
