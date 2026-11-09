import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def training_box(annotation, mask, width, height):
    """Keep positive thin boxes; repair degenerate boxes from the actual mask only."""
    x, y, w, h = annotation["bbox"]
    box = [
        max(0.0, float(x)),
        max(0.0, float(y)),
        min(float(width), float(x + w)),
        min(float(height), float(y + h)),
    ]
    repaired = box[2] <= box[0] or box[3] <= box[1]
    if repaired:
        ys, xs = np.where(mask)
        if not len(xs):
            raise ValueError(
                f"Annotation {annotation['id']} has no valid box or mask pixels."
            )
        box = [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        ]
    return box, repaired


class WireDataset(Dataset):
    def __init__(
        self,
        annotations,
        images,
        *,
        training=False,
        include_targets=True,
        horizontal_flip_probability=0.5,
        max_images=None,
    ):
        self.annotation_path = Path(annotations)
        self.images_root = Path(images)
        self.coco = COCO(str(self.annotation_path))
        self.image_ids = sorted(
            self.coco.getImgIds()
        )  # Includes images without annotations.
        if max_images is not None:
            if max_images < 1:
                raise ValueError("max_images must be positive.")
            self.image_ids = self.image_ids[:max_images]
        self.category_ids = sorted(self.coco.getCatIds())
        if (
            len(self.category_ids) != 1
            or self.coco.cats[self.category_ids[0]]["name"] != "cable"
        ):
            raise ValueError("Expected exactly one foreground category named cable.")
        # Torchvision reserves 0 for background. COCO's category 0 becomes model label 1.
        self.category_to_label = {
            category: i + 1 for i, category in enumerate(self.category_ids)
        }
        self.label_to_category = {
            label: category for category, label in self.category_to_label.items()
        }
        self.training = training
        self.include_targets = include_targets
        self.flip_probability = horizontal_flip_probability

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        info = self.coco.imgs[image_id]
        with Image.open(self.images_root / info["file_name"]) as source:
            source = source.convert("RGB")
            if source.size != (info["width"], info["height"]):
                raise ValueError(
                    "Image and annotation dimensions disagree: " + info["file_name"]
                )
            image = pil_to_tensor(source).to(torch.float32) / 255.0
        if not self.include_targets:
            return image, {
                "image_id": image_id,
                "width": info["width"],
                "height": info["height"],
            }
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        boxes, labels, masks, area, crowd, repaired_ids, empty_mask_ids = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for annotation in annotations:
            mask = self.coco.annToMask(annotation).astype(np.uint8)
            # Subpixel polygons can rasterize to all zeros. Retain their box supervision
            # and original rasterized mask, rather than dropping instances or inventing pixels.
            if not mask.any():
                empty_mask_ids.append(annotation["id"])
            box, repaired = training_box(
                annotation, mask, info["width"], info["height"]
            )
            boxes.append(box)
            labels.append(self.category_to_label[annotation["category_id"]])
            masks.append(mask)
            area.append(float(mask.sum()))
            crowd.append(annotation.get("iscrowd", 0))
            if repaired:
                repaired_ids.append(annotation["id"])
        if any(crowd):
            raise ValueError(
                "Crowd annotations require a separate training policy; none are expected here."
            )
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.from_numpy(
                np.stack(masks)
                if masks
                else np.zeros((0, info["height"], info["width"]), dtype=np.uint8)
            ),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(crowd, dtype=torch.int64),
            "repaired_annotation_ids": torch.tensor(repaired_ids, dtype=torch.int64),
            "empty_mask_annotation_ids": torch.tensor(
                empty_mask_ids, dtype=torch.int64
            ),
        }
        if self.training and random.random() < self.flip_probability:
            image = image.flip(-1)
            target["masks"] = target["masks"].flip(-1)
            old = target["boxes"].clone()
            target["boxes"][:, 0] = info["width"] - old[:, 2]
            target["boxes"][:, 2] = info["width"] - old[:, 0]
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
