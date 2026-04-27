from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .config import EMOTION_CLASSES, MODEL_PATH, PROJECT_ROOT
from .data_summary import count_image_dataset, totals_by_emotion, totals_by_split

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
        data_frame.columnconfigure(0, weight=1)

        columns = ("emotion", "train", "val", "test", "total")
        self.table = ttk.Treeview(data_frame, columns=columns, show="headings", height=10)
        for column in columns:
            self.table.heading(column, text=column.title())
            self.table.column(column, anchor="center", width=120)
        self.table.grid(row=0, column=0, sticky="nsew")

        status_frame = ttk.LabelFrame(root, text="Environment Status", padding=12)
        status_frame.grid(row=1, column=1, sticky="nsew", pady=(18, 12), padx=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        self.status_text = tk.Text(status_frame, height=13, wrap="word", relief="flat")
        self.status_text.grid(row=0, column=0, sticky="nsew")

        actions = ttk.Frame(root)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        for i in range(5):
            actions.columnconfigure(i, weight=1)

        ttk.Button(actions, text="Refresh", command=self._refresh_summary).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(actions, text="Run Camera Demo", command=self._run_camera_demo).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(actions, text="Train 1-Epoch Smoke Test", command=self._run_train_smoke).grid(row=0, column=2, sticky="ew", padx=4)
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

    def _run_train_smoke(self) -> None:
        self._run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "train" / "train_emotion.py"),
                "--epochs",
                "1",
                "--limit-train",
                "256",
                "--limit-val",
                "128",
            ],
            "Training smoke test",
        )

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

