"""Audit local selection and export a readable source index without changing labels."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import numpy as np
from PIL import Image, ImageOps
from wirelab.io import ROOT, manifest, sha, read, write, MaskStore
from prepare_local_dataset import RAW, nadir_status


def pixel_hash(path):
    with Image.open(path) as image:
        pixels = np.asarray(
            ImageOps.exif_transpose(image)
            .convert("RGB")
            .resize((700, 700), Image.Resampling.LANCZOS)
        )
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def main():
    m = manifest()
    items = m["images"]
    store = MaskStore()
    assert m["local_only"]
    assert len(items) == len({i["key"] for i in items})
    assert len(items) == len({i["image_sha256"] for i in items})
    raw_paths = [
        p for p in RAW.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    raw_bytes = {sha(p) for p in raw_paths}
    raw_pixels = {pixel_hash(p) for p in raw_paths}
    selected_pixels = {}
    index = []
    for i in items:
        assert i["source"] == "local" and i["file_name"].startswith("DJI")
        assert nadir_status(i["camera_metadata"]) == "eligible"
        path = ROOT / i["image_path"]
        assert sha(path) == i["image_sha256"]
        digest = pixel_hash(path)
        assert digest not in selected_pixels, (i["key"], selected_pixels.get(digest))
        selected_pixels[digest] = i["key"]
        assert not (i["role"] == "test" and i.get("segmentation_training_seen", False))
        if i["original_split"] == "external":
            source = Path(i["source_path"])
            assert (
                sha(source) == i["source_sha256"]
                and i["source_sha256"] not in raw_bytes
            )
            assert pixel_hash(source) not in raw_pixels
        if i.get("has_gt_mask", True):
            store.load(i, "gt")
        assert i["key"] in store.pred, ("Missing prediction", i["key"])
        store.load(i, "pred")
        index.append(
            dict(
                key=i["key"],
                role=i["role"],
                original_split=i["original_split"],
                source_file=i["file_name"],
                source_path=i["source_path"],
                image_path=str(path),
                gimbal_pitch_deg=i["camera_metadata"]["GimbalPitchDegree"],
                gimbal_roll_deg=i["camera_metadata"]["GimbalRollDegree"],
                segmentation_training_seen=i.get("segmentation_training_seen", False),
                has_gt_mask=i.get("has_gt_mask", True),
                image_sha256=i["image_sha256"],
            )
        )
    assert {
        p.resolve() for p in (ROOT / "dataset/images").iterdir() if p.is_file()
    } == {(ROOT / i["image_path"]).resolve() for i in items}
    for split in ("train", "valid", "test"):
        doc = read(ROOT / f"dataset/coco_{split}.json")
        assert all(i["file_name"].startswith("DJI") for i in doc["images"])
    labels = list((ROOT / "annotations").glob("*.json"))
    assert all(p.stem in {i["key"] for i in items} for p in labels)
    for split, digest in m["input_annotation_hashes"].items():
        assert (
            sha(
                ROOT.parent
                / "segmentation/rfdetr/dataset_new"
                / split
                / "_annotations.coco.json"
            )
            == digest
        )
    with (ROOT / "dataset/sample_index.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    report = dict(
        images=len(items),
        roles=dict(Counter(i["role"] for i in items)),
        sources=dict(Counter(i["source"] for i in items)),
        original_splits=dict(Counter(i["original_split"] for i in items)),
        prediction_cache_complete=True,
        gt_mask_images=sum(i.get("has_gt_mask", True) for i in items),
        active_human_annotation_files=len(labels),
        external_source_exact_duplicates_of_originals=0,
        normalized_pixel_duplicates=0,
        original_annotation_files_unchanged=True,
        passed=True,
    )
    write(ROOT / "validation/local_dataset_check.json", report)
    print(report)


if __name__ == "__main__":
    main()
