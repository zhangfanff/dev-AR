"""Launch the wire-direction annotation tool."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "direction"))
from annotate_angles import main

if __name__ == "__main__":
    main()
