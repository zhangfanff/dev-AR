"""Locate external model frameworks used by the project scripts."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent


def activate(name, required=True):
    key = "INSPECTION_" + name.upper() + "_ROOT"
    folder = Path(os.environ.get(key, ROOT / ".external" / name)).resolve()
    if not folder.is_dir():
        if not required:
            return None
        raise RuntimeError(
            f"Missing {name} source. Run: python third_party/setup_dependencies.py --only {name}"
        )
    sys.path.insert(0, str(folder))
    return folder
