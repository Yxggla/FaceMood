from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .config import EMOTION_CLASSES, IMAGE_DATA_DIR, PROJECT_ROOT
from .data_summary import count_image_dataset, totals_by_emotion, totals_by_split


def build_dataset_summary(data_dir: Path = IMAGE_DATA_DIR) -> dict:
    counts = count_image_dataset(data_dir)
    split_totals = totals_by_split(counts)
    emotion_totals = totals_by_emotion(counts)
    total = sum(split_totals.values())
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "classes": EMOTION_CLASSES,
        "counts": counts,
        "split_totals": split_totals,
        "emotion_totals": emotion_totals,
        "total_images": total,
        "notes": [
            "FER2013 is used in the 7-class ImageFolder layout.",
            "The disgust class is much smaller than the other classes, so final training should discuss class imbalance.",
        ],
    }


def write_dataset_summary(output_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    payload = build_dataset_summary()
    metrics_dir = output_root / "results" / "metrics"
    report_dir = output_root / "report"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = metrics_dir / "dataset_summary.json"
    csv_path = metrics_dir / "dataset_summary.csv"
    md_path = report_dir / "DATASET_SUMMARY.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, payload)
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def _write_csv(path: Path, payload: dict) -> None:
    counts = payload["counts"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["emotion", "train", "val", "test", "total"])
        for emotion in EMOTION_CLASSES:
            writer.writerow(
                [
                    emotion,
                    counts["train"][emotion],
                    counts["val"][emotion],
                    counts["test"][emotion],
                    payload["emotion_totals"][emotion],
                ]
            )
        writer.writerow(
            [
                "TOTAL",
                payload["split_totals"]["train"],
                payload["split_totals"]["val"],
                payload["split_totals"]["test"],
                payload["total_images"],
            ]
        )


def _to_markdown(payload: dict) -> str:
    counts = payload["counts"]
    lines = [
        "# FER2013 Dataset Summary",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        f"Data source: `{payload['data_dir']}`",
        "",
        "## Class Counts",
        "",
        "| Emotion | Train | Validation | Test | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for emotion in EMOTION_CLASSES:
        lines.append(
            f"| {emotion} | {counts['train'][emotion]} | {counts['val'][emotion]} | "
            f"{counts['test'][emotion]} | {payload['emotion_totals'][emotion]} |"
        )
    lines.extend(
        [
            f"| **TOTAL** | **{payload['split_totals']['train']}** | "
            f"**{payload['split_totals']['val']}** | **{payload['split_totals']['test']}** | "
            f"**{payload['total_images']}** |",
            "",
            "## Notes for Report",
            "",
            "- The project uses all 7 FER2013 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise.",
            "- The `disgust` class is heavily underrepresented compared with the other classes.",
            "- This imbalance should be mentioned in the findings/discussion section after model evaluation.",
            "",
        ]
    )
    return "\n".join(lines)

