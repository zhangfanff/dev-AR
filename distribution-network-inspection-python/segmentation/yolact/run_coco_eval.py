"""Run the yolact run_coco_eval.py entry point."""

from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dependencies import activate

if __name__ == "__main__":
    folder = activate("yolact")
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    runpy.run_path(str(folder / "run_coco_eval.py"), run_name="__main__")
