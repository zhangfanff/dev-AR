"""Prepare original YOLOv5 v7.0 source and weights without starting training."""

from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

PROJECT = Path(__file__).resolve().parent
VENDOR = PROJECT / "baseline_scripts/vendor/yolov5"


def download(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Python detector baseline setup"}
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def main():
    if not VENDOR.exists():
        archive = zipfile.ZipFile(
            io.BytesIO(
                download(
                    "https://codeload.github.com/ultralytics/yolov5/zip/refs/tags/v7.0"
                )
            )
        )
        for member in archive.infolist():
            parts = Path(member.filename).parts[1:]
            if not parts:
                continue
            path = VENDOR.joinpath(*parts)
            if not path.resolve().is_relative_to(VENDOR.resolve()):
                raise ValueError("Invalid archive path")
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(member))
    # Minimal data-policy additions: unchanged network, losses, optimizer and matching.
    patches = [
        (
            "train.py",
            "        callbacks.run('on_train_epoch_start')",
            "        # PAPER_BASELINE_CLOSE_MOSAIC\n        if int(hyp.get('close_mosaic', 0)) > 0 and epoch == max(epochs - int(hyp['close_mosaic']), 0):\n            dataset.mosaic = False\n            # Recreate workers so they receive the updated dataset policy.\n            if hasattr(train_loader, 'iterator'):\n                train_loader.iterator = torch.utils.data.DataLoader.__iter__(train_loader)\n        callbacks.run('on_train_epoch_start')",
        ),
        (
            "utils/dataloaders.py",
            "        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB",
            "        # PAPER_BASELINE_BGR\n        if self.augment and random.random() < hyp.get('bgr', 0.0):\n            img = img[..., ::-1]\n        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB",
        ),
    ]
    for filename, old, new in patches:
        path = VENDOR / filename
        text = path.read_text(encoding="utf-8")
        marker = (
            "PAPER_BASELINE_CLOSE_MOSAIC"
            if filename == "train.py"
            else "PAPER_BASELINE_BGR"
        )
        previous = "        # PAPER_BASELINE_CLOSE_MOSAIC\n        if epoch >= epochs - int(hyp.get('close_mosaic', 0)):\n            dataset.mosaic = False\n        callbacks.run('on_train_epoch_start')"
        if filename == "train.py" and previous in text:
            text = text.replace(previous, new)
            path.write_text(text, encoding="utf-8")
        if marker not in text:
            if text.count(old) != 1:
                raise RuntimeError(f"Unexpected official source: {path}")
            path.write_text(text.replace(old, new), encoding="utf-8")
    weights = PROJECT / "yolov5n.pt"
    if not weights.exists():
        payload = download(
            "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
        )
        if len(payload) < 100000:
            raise ValueError("Downloaded file is too small to be weights")
        weights.write_bytes(payload)
    manifest = {
        "source": "https://github.com/ultralytics/yolov5/tree/v7.0",
        "version": "v7.0",
        "model": "original anchor-based YOLOv5n; not YOLOv5u",
        "weights": str(weights),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "local_patches": [
            "last-ten-epoch Mosaic shutdown",
            "task-specific BGR reversal probability",
        ],
        "compatibility": [
            "torch.load(weights_only=False) only in isolated v5 runtime for trusted checkpoints",
            "Pillow getsize compatibility",
            "Albumentations 2.x equivalent active transforms",
        ],
    }
    (VENDOR.parent / "yolov5_provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
