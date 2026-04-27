from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from facemood.reporting import write_dataset_summary


def main() -> None:
    outputs = write_dataset_summary(ROOT)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

