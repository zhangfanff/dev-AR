"""Run the rfdetr_compat eval.py entry point."""

from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dependencies import activate

if __name__ == "__main__":
    folder = activate("rfdetr_compat")
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    runpy.run_path(str(folder / "eval.py"), run_name="__main__")
