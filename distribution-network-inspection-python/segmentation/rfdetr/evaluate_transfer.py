"""Evaluate segmentation transfer on local distribution-line images."""

from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json
import sys

import numpy as np
from PIL import Image
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from coco_evaluation import evaluate_predictions


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_and_subset(data_root, out):
    # Use the actual training export. Its local-source provenance was audited earlier.
    full = load_json(data_root / "test/_annotations.coco.json")
    local = [i for i in full["images"] if i["file_name"].startswith("DJI")]
    assert local
    ids = {i["id"] for i in local}
    sub = copy.deepcopy(full)
    sub["images"] = local
    sub["annotations"] = [a for a in full["annotations"] if a["image_id"] in ids]
    sub.setdefault("info", {})
    assert {a["category_id"] for a in sub["annotations"]} == {0}
    hashes = {
        i["file_name"]: sha256(data_root / "test" / i["file_name"]) for i in local
    }
    local_hashes = set(hashes.values())
    checks = {}
    for split, rf_split in [("train", "train"), ("val", "valid")]:
        mixed_root = data_root / rf_split
        d = load_json(mixed_root / "_annotations.coco.json")
        hs = {sha256(mixed_root / i["file_name"]) for i in d["images"]}
        overlap = sorted(local_hashes & hs)
        assert not overlap, (split, overlap)
        checks["mixed_" + split + "_exact_image_overlap"] = len(overlap)
        baseline_root = PROJECT / "dataset_ttpla" / rf_split
        d = load_json(baseline_root / "_annotations.coco.json")
        hs = {sha256(baseline_root / i["file_name"]) for i in d["images"]}
        overlap = sorted(local_hashes & hs)
        assert not overlap, ("source_" + split, overlap)
        checks["source_" + split + "_exact_image_overlap"] = len(overlap)
    target = out / "local_test_annotations.json"
    target.write_text(json.dumps(sub), encoding="utf-8")
    report = {
        "images": len(local),
        "instances": len(sub["annotations"]),
        "image_ids": sorted(ids),
        "local_selector": "DJI filename prefix, consistent with previous author-source provenance audit",
        "image_sha256": hashes,
        "checks": checks,
        "scope": "Exact-image disjointness; no route/session independence or near-duplicate claim.",
    }
    (out / "holdout_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("HOLDOUT", json.dumps(report), flush=True)
    return target, sub


def evaluate_checkpoint(
    name, checkpoint, resolution, annotations, sub, images, out, score
):
    from rfdetr import RFDETRSegPreview

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training_args = vars(ck["args"])
    tracked = (
        "resolution",
        "num_classes",
        "class_names",
        "epochs",
        "batch_size",
        "grad_accum_steps",
        "lr",
        "lr_encoder",
        "seed",
        "num_select",
        "num_queries",
    )
    config = {k: training_args.get(k) for k in tracked}
    del ck
    model = RFDETRSegPreview(
        pretrain_weights=str(checkpoint),
        resolution=resolution,
        num_select=200,
        device="cuda",
    )
    assert model.model.resolution == resolution
    assert model.model.class_names[0] == "cable"
    predictions = {"bbox": [], "segm": []}
    for info in sorted(sub["images"], key=lambda x: x["id"]):
        with Image.open(images / info["file_name"]) as image:
            pixels = np.array(image.convert("RGB"))
        with torch.inference_mode():
            pred = model.predict(pixels, threshold=score)
        cable = np.flatnonzero(pred.class_id == 0)
        order = cable[np.argsort(-pred.confidence[cable], kind="stable")[:100]]
        for j in order:
            x1, y1, x2, y2 = map(float, pred.xyxy[j])
            common = {
                "image_id": info["id"],
                "category_id": 0,
                "score": float(pred.confidence[j]),
            }
            predictions["bbox"].append(dict(common, bbox=[x1, y1, x2 - x1, y2 - y1]))
            mask = np.asarray(pred.mask[j])
            assert mask.shape == (info["height"], info["width"])
            rle = mask_utils.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            predictions["segm"].append(dict(common, segmentation=rle))
        print(name, resolution, info["id"], "cable_predictions", len(order), flush=True)
    for kind, rows in predictions.items():
        (
            out / (name + "_" + str(resolution) + "_" + kind + "_predictions.json")
        ).write_text(json.dumps(rows), encoding="utf-8")
    gt = COCO(str(annotations))
    metrics = evaluate_predictions(gt, predictions, sorted(gt.getImgIds()), 100)
    result = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "training_config": config,
        "inference_resolution": resolution,
        "inference_num_select": 200,
        "score_threshold": score,
        "max_cable_detections": 100,
        "image_count": len(sub["images"]),
        "gt_instances": len(sub["annotations"]),
        "annotation_sha256": sha256(annotations),
        "torch": torch.__version__,
        "rfdetr": importlib.metadata.version("rfdetr"),
        "gpu": torch.cuda.get_device_name(0),
        "metrics": metrics,
        "interpretation": "Existing-checkpoint target-domain comparison; training resolution, class set and batch budget differ. Not an isolated data ablation.",
    }
    (out / (name + "_" + str(resolution) + "_metrics.json")).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("RESULT", name, resolution, json.dumps(metrics), flush=True)
    del model
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=PROJECT / "dataset_new")
    p.add_argument(
        "--out", type=Path, default=PROJECT / "results/revision5_transfer_20261102"
    )
    p.add_argument("--resolution", type=int, default=648)
    p.add_argument("--score", type=float, default=0.3)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    annotations, sub = audit_and_subset(args.data_root, args.out)
    results = []
    for name, folder in [
        ("TTPLA_only", "model_ttpla432"),
        ("TTPLA_plus_local", "model_new648"),
    ]:
        results.append(
            evaluate_checkpoint(
                name,
                PROJECT / folder / "checkpoint_best_total.pth",
                args.resolution,
                annotations,
                sub,
                args.data_root / "test",
                args.out,
                args.score,
            )
        )
    (args.out / "comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
