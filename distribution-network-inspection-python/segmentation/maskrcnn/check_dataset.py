"""Audit the exact train/val/test files without re-splitting or changing annotations."""

import argparse
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO

from wireseg.common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    read_config,
    sha256,
    split_paths,
    write_json,
)
from wireseg.dataset import training_box


def audit(config, *, check_masks=False, check_hashes=False):
    report = {"dataset_root": config["dataset_root"], "splits": {}}
    names, hashes = {}, {}
    for split in ("train", "val", "test"):
        annotations, images = split_paths(config, split)
        coco = COCO(str(annotations))
        if list(coco.cats) != [0] or coco.cats[0]["name"] != "cable":
            raise ValueError("Expected single cable category with COCO ID 0.")
        names[split], hashes[split] = set(), set()
        dimensions = set()
        for info in coco.imgs.values():
            path = images / info["file_name"]
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                if image.size != (info["width"], info["height"]):
                    raise ValueError(f"Annotation/image dimension mismatch: {path}")
                dimensions.add(image.size)
            names[split].add(info["file_name"])
            if check_hashes:
                hashes[split].add(sha256(path))
        degenerate, empty_masks = [], []
        for annotation in coco.anns.values():
            if annotation["bbox"][2] <= 0 or annotation["bbox"][3] <= 0:
                mask = coco.annToMask(annotation)
                repaired, _ = training_box(
                    annotation,
                    mask,
                    coco.imgs[annotation["image_id"]]["width"],
                    coco.imgs[annotation["image_id"]]["height"],
                )
                degenerate.append(
                    {
                        "annotation_id": annotation["id"],
                        "original_bbox": annotation["bbox"],
                        "training_xyxy": repaired,
                    }
                )
            if annotation.get("iscrowd", 0):
                raise ValueError(
                    "Crowd annotation found; current training policy expects no crowd."
                )
            if check_masks and not coco.annToMask(annotation).any():
                empty_masks.append(annotation["id"])
        report["splits"][split] = {
            "images": len(coco.imgs),
            "instances": len(coco.anns),
            "negative_images": sum(not coco.getAnnIds(imgIds=[i]) for i in coco.imgs),
            "dimensions": sorted(dimensions),
            "annotations": str(annotations.resolve()),
            "annotation_sha256": sha256(annotations),
            "degenerate_boxes": degenerate,
            "empty_rasterized_mask_ids": empty_masks if check_masks else None,
        }
        print(split, report["splits"][split], flush=True)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if names[left] & names[right]:
            raise ValueError(f"Overlapping image names: {left}/{right}")
        if check_hashes and hashes[left] & hashes[right]:
            raise ValueError(f"Byte-identical images across splits: {left}/{right}")
    report["all_masks_checked"] = check_masks
    report["image_hashes_checked"] = check_hashes
    report["note"] = (
        "Original annotations are unchanged. Degenerate training boxes are recovered from masks. "
        "Subpixel polygons with empty rasterized masks retain box labels and original zero masks; "
        "evaluation always uses unmodified original COCO annotations."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check-masks", action="store_true")
    parser.add_argument("--check-hashes", action="store_true")
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "results/dataset_audit.json")
    )
    args = parser.parse_args()
    report = audit(
        read_config(args.config),
        check_masks=args.check_masks,
        check_hashes=args.check_hashes,
    )
    write_json(args.output, report)
    print("PASS: exact splits audited; saved", args.output)


if __name__ == "__main__":
    main()
