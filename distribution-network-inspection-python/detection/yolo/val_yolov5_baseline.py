import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline_scripts"))
from val_v5 import main

if __name__ == "__main__":
    main("yolov5n")
