"""Load original YOLOv5 in its own process; never mix with ultralytics modules."""

import os
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[3] / ".external/yolov5"


def activate():
    if not (VENDOR / "models/yolo.py").is_file():
        raise FileNotFoundError(
            f"YOLOv5 source missing; run third_party/setup_dependencies.py --only yolov5: {VENDOR}"
        )
    sys.path.insert(0, str(VENDOR))
    # v7.0 uses os.path.relpath(ROOT, cwd), which fails across Windows drives.
    # All caller data/weights/output paths are resolved before this change.
    os.chdir(VENDOR)
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    import torch

    # YOLOv5 v7.0 checkpoints serialize model classes. Apply only in this
    # isolated process, to official pretrained or locally trained weights.
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    # Pillow >=10 removed getsize; the original plotting code still uses it.
    from PIL import ImageFont

    for font in [ImageFont.ImageFont, ImageFont.FreeTypeFont]:
        if not hasattr(font, "getsize"):

            def getsize(self, text):
                x0, y0, x1, y1 = self.getbbox(text)
                return x1 - x0, y1 - y0

            font.getsize = getsize
    # The official wrapper uses Albumentations 1.x names for transforms whose
    # probability is zero. Keep the active four transforms with the 2.x API.
    import albumentations as A

    if int(A.__version__.split(".")[0]) >= 2:
        from utils.augmentations import Albumentations

        def init_albumentations(self, size=640):
            transforms = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
            ]
            self.transform = A.Compose(
                transforms,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

        Albumentations.__init__ = init_albumentations
    return VENDOR
