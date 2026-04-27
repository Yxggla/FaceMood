from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .config import EMOTION_CLASSES, IMAGE_DATA_DIR, MODEL_PATH, PROJECT_ROOT
from .data_summary import count_image_dataset, totals_by_emotion, totals_by_split
from .reporting import write_dataset_summary

OPTIONAL_DEPENDENCIES = {
    "OpenCV camera/display": "cv2",
    "NumPy arrays": "numpy",
    "PyTorch model": "torch",
    "TorchVision dataset": "torchvision",
    "MediaPipe landmarks": "mediapipe",
    "Matplotlib charts": "matplotlib",
    "Scikit-learn metrics": "sklearn",
    "Pillow CSV converter": "PIL",
}


class FaceMoodLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FaceMood Project Launcher")
        self.geometry("980x680")
        self.minsize(900, 620)
        self.configure(bg="#f7f7f4")
        self.output = tk.StringVar(value="Ready. This launcher works without the emotion model.")
        self.sample_images: list[tk.PhotoImage] = []
        self._build()
        self._refresh_summary()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(1, weight=1)

        title = ttk.Label(root, text="FaceMood", font=("Arial", 26, "bold"))
        title.grid(row=0, column=0, sticky="w")
        subtitle = ttk.Label(
            root,
            text="Real-time face detection, landmarks, and 7-class FER2013 emotion recognition",
            font=("Arial", 12),
        )
        subtitle.grid(row=0, column=1, sticky="e")

        data_frame = ttk.LabelFrame(root, text="Dataset Overview", padding=12)
        data_frame.grid(row=1, column=0, sticky="nsew", pady=(18, 12), padx=(0, 10))
        data_frame.rowconfigure(0, weight=1)
        data_frame.rowconfigure(1, weight=1)
        data_frame.columnconfigure(0, weight=1)

        columns = ("emotion", "train", "val", "test", "total")
        self.table = ttk.Treeview(data_frame, columns=columns, show="headings", height=10)
        for column in columns:
            self.table.heading(column, text=column.title())
            self.table.column(column, anchor="center", width=120)
        self.table.grid(row=0, column=0, sticky="nsew")
        self.chart = tk.Canvas(data_frame, height=180, bg="#ffffff", highlightthickness=1, highlightbackground="#d6d6d0")
        self.chart.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        status_frame = ttk.LabelFrame(root, text="Environment Status", padding=12)
        status_frame.grid(row=1, column=1, sticky="nsew", pady=(18, 12), padx=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        status_frame.rowconfigure(1, weight=1)
        self.status_text = tk.Text(status_frame, height=13, wrap="word", relief="flat")
        self.status_text.grid(row=0, column=0, sticky="nsew")
        self.sample_frame = ttk.Frame(status_frame)
        self.sample_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        actions = ttk.Frame(root)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        for i in range(5):
            actions.columnconfigure(i, weight=1)

        ttk.Button(actions, text="Refresh", command=self._refresh_summary).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(actions, text="Run Camera Demo", command=self._run_camera_demo).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(actions, text="Generate Dataset Report", command=self._generate_dataset_report).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(actions, text="Open README", command=lambda: self._open_path(PROJECT_ROOT / "README.md")).grid(
            row=0, column=3, sticky="ew", padx=4
        )
        ttk.Button(actions, text="Open Project Folder", command=lambda: self._open_path(PROJECT_ROOT)).grid(
            row=0, column=4, sticky="ew", padx=4
        )

        output_frame = ttk.LabelFrame(root, text="Launcher Output", padding=12)
        output_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        output = ttk.Label(output_frame, textvariable=self.output, anchor="w", justify="left")
        output.pack(fill="x")

    def _refresh_summary(self) -> None:
        summary = count_image_dataset()
        split_totals = totals_by_split(summary)
        emotion_totals = totals_by_emotion(summary)
        for item in self.table.get_children():
            self.table.delete(item)
        for emotion in EMOTION_CLASSES:
            self.table.insert(
                "",
                "end",
                values=(
                    emotion,
                    summary["train"][emotion],
                    summary["val"][emotion],
                    summary["test"][emotion],
                    emotion_totals[emotion],
                ),
        )
        self.table.insert("", "end", values=("TOTAL", split_totals["train"], split_totals["val"], split_totals["test"], sum(split_totals.values())))
        self._draw_chart(emotion_totals)
        self._load_samples()
        self._refresh_environment()

    def _refresh_environment(self) -> None:
        lines = [
            f"Python: {sys.version.split()[0]}",
            f"Project: {PROJECT_ROOT}",
            f"Model: {MODEL_PATH}",
            f"Model exists: {'yes' if MODEL_PATH.exists() else 'no - camera demo will show unknown labels'}",
            "",
            "Optional dependencies:",
        ]
        for label, module in OPTIONAL_DEPENDENCIES.items():
            found = importlib.util.find_spec(module) is not None
            lines.append(f"  [{'ok' if found else 'missing'}] {label} ({module})")
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", "\n".join(lines))
        self.status_text.configure(state="disabled")
        self.output.set("Dataset and environment status refreshed.")

    def _run_camera_demo(self) -> None:
        self._run_command([sys.executable, str(PROJECT_ROOT / "src" / "main.py")], "Camera demo")

    def _generate_dataset_report(self) -> None:
        outputs = write_dataset_summary(PROJECT_ROOT)
        rendered = "\n".join(f"{name}: {path}" for name, path in outputs.items())
        self.output.set(f"Dataset report generated.\n{rendered}")

    def _draw_chart(self, emotion_totals: dict[str, int]) -> None:
        self.chart.delete("all")
        width = max(720, self.chart.winfo_width())
        height = 180
        max_count = max(emotion_totals.values()) if emotion_totals else 1
        margin_x = 28
        baseline = height - 34
        bar_area = width - margin_x * 2
        bar_width = max(28, int(bar_area / len(EMOTION_CLASSES) * 0.58))
        step = bar_area / len(EMOTION_CLASSES)
        colors = ["#d84a3a", "#4b8f48", "#7b5fc8", "#d7a729", "#7f858c", "#3f7fb5", "#d9862f"]
        self.chart.create_text(12, 10, text="Total images by emotion", anchor="nw", fill="#262626", font=("Arial", 11, "bold"))
        self.chart.create_line(margin_x, baseline, width - margin_x, baseline, fill="#9a9a94")
        for index, emotion in enumerate(EMOTION_CLASSES):
            count = emotion_totals[emotion]
            x_center = margin_x + step * index + step / 2
            bar_height = int((count / max_count) * 104)
            x1 = int(x_center - bar_width / 2)
            x2 = int(x_center + bar_width / 2)
            y1 = baseline - bar_height
            color = colors[index % len(colors)]
            self.chart.create_rectangle(x1, y1, x2, baseline, fill=color, outline="")
            self.chart.create_text(x_center, y1 - 8, text=str(count), fill="#262626", font=("Arial", 9))
            self.chart.create_text(x_center, baseline + 14, text=emotion, fill="#333333", font=("Arial", 9))

    def _load_samples(self) -> None:
        for child in self.sample_frame.winfo_children():
            child.destroy()
        self.sample_images = []
        ttk.Label(self.sample_frame, text="Sample images", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        for index, emotion in enumerate(EMOTION_CLASSES):
            image_path = _first_sample_path(emotion)
            row = 1 + index // 4
            column = index % 4
            cell = ttk.Frame(self.sample_frame, padding=4)
            cell.grid(row=row, column=column, sticky="n")
            if image_path is not None:
                try:
                    image = tk.PhotoImage(file=str(image_path)).zoom(2, 2)
                    self.sample_images.append(image)
                    ttk.Label(cell, image=image).pack()
                except tk.TclError:
                    ttk.Label(cell, text="no preview", width=12).pack()
            else:
                ttk.Label(cell, text="missing", width=12).pack()
            ttk.Label(cell, text=emotion, font=("Arial", 9)).pack()

    def _run_command(self, command: list[str], label: str) -> None:
        self.output.set(f"{label} started. If dependencies are missing, install requirements.txt first.")
        threading.Thread(target=self._run_command_worker, args=(command, label), daemon=True).start()

    def _run_command_worker(self, command: list[str], label: str) -> None:
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            message = completed.stdout.strip() or completed.stderr.strip() or f"{label} finished."
            if completed.returncode != 0:
                message = f"{label} failed with exit code {completed.returncode}.\n{message}"
        except Exception as exc:  # pragma: no cover - UI safety net
            message = f"{label} failed: {exc}"
        self.after(0, lambda: self.output.set(message[:1400]))

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            elif os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))


def main() -> None:
    app = FaceMoodLauncher()
    app.mainloop()


def _first_sample_path(emotion: str) -> Path | None:
    for split in ["train", "val", "test"]:
        emotion_dir = IMAGE_DATA_DIR / split / emotion
        if not emotion_dir.exists():
            continue
        for path in sorted(emotion_dir.glob("*.png")):
            return path
    return None
