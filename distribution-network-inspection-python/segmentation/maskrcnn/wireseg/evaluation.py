import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader

from .coco_metrics import evaluate_predictions, print_per_class
from .common import environment, sha256, write_json
from .dataset import collate_fn


def evaluation_identity(dataset, config):
    ids = dataset.image_ids
    selected = set(ids)
    return {
        "protocol_version": 1,
        "model": "maskrcnn_resnet50_fpn",
        "annotations": str(dataset.annotation_path.resolve()),
        "annotation_sha256": sha256(dataset.annotation_path),
        "images": str(dataset.images_root.resolve()),
        "image_ids": ids,
        "image_count": len(ids),
        "gt_instance_count": sum(
            a["image_id"] in selected for a in dataset.coco.anns.values()
        ),
        "empty_gt_images": sum(not dataset.coco.getAnnIds(imgIds=[i]) for i in ids),
        "label_to_category": {str(k): v for k, v in dataset.label_to_category.items()},
        "input_size": config["model"]["input_size"],
        "nms_threshold": config["model"]["nms_threshold"],
        **config["evaluation"],
    }


def prediction_paths(output):
    root = Path(output)
    return {
        kind: root / f"maskrcnn_{kind}_predictions.json" for kind in ("bbox", "segm")
    }


def score_and_save(dataset, predictions, metadata, output):
    output = Path(output)
    metrics = evaluate_predictions(
        dataset.coco, predictions, dataset.image_ids, max_detections=100
    )
    print_per_class(metrics)
    report = dict(metadata, metrics=metrics)
    write_json(output / "maskrcnn_metrics.json", report)
    return report


@torch.inference_mode()
def evaluate_model(model, dataset, config, device, output, *, metadata=None, workers=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = evaluation_identity(dataset, config)
    provenance = dict(identity, **(metadata or {}), environment=environment())
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    predictions = {"bbox": [], "segm": []}
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        for index, (images, infos) in enumerate(loader):
            result = model([image.to(device, non_blocking=True) for image in images])[0]
            info = infos[0]
            ranking = torch.argsort(result["scores"], descending=True, stable=True)
            ranking = ranking[
                result["scores"][ranking] >= config["evaluation"]["score_threshold"]
            ][:100]
            for j in ranking.cpu().tolist():
                label = int(result["labels"][j].item())
                if label not in dataset.label_to_category:
                    raise ValueError(f"Unexpected foreground label {label}.")
                category = dataset.label_to_category[label]
                x1, y1, x2, y2 = result["boxes"][j].cpu().tolist()
                score = float(result["scores"][j].item())
                predictions["bbox"].append(
                    {
                        "image_id": info["image_id"],
                        "category_id": category,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
                mask = (
                    result["masks"][j, 0].cpu().numpy()
                    >= config["evaluation"]["mask_threshold"]
                )
                if mask.shape != (info["height"], info["width"]):
                    raise ValueError(
                        "Prediction mask is not at original image resolution."
                    )
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                predictions["segm"].append(
                    {
                        "image_id": info["image_id"],
                        "category_id": category,
                        "segmentation": rle,
                        "score": score,
                    }
                )
            if index % 25 == 0 or index + 1 == len(dataset):
                print(
                    f"COCO inference {index + 1}/{len(dataset)} ({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    finally:
        model.train(was_training)
    provenance["prediction_count"] = len(predictions["bbox"])
    provenance["prediction_sha256"] = {}
    for kind, path in prediction_paths(output).items():
        write_json(path, predictions[kind])
        provenance["prediction_sha256"][kind] = sha256(path)
    write_json(output / "maskrcnn_metadata.json", provenance)
    print(
        f"Evaluation population: {identity['image_count']} images, "
        f"{identity['gt_instance_count']} GT, {identity['empty_gt_images']} negative images."
    )
    return score_and_save(dataset, predictions, provenance, output)


def reuse_predictions(dataset, config, output, expected_metadata):
    import json

    output = Path(output)
    metadata = json.loads(
        (output / "maskrcnn_metadata.json").read_text(encoding="utf-8")
    )
    expected = dict(evaluation_identity(dataset, config), **expected_metadata)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("Saved predictions have a different protocol: " + key)
    predictions = {}
    for kind, path in prediction_paths(output).items():
        if sha256(path) != metadata["prediction_sha256"][kind]:
            raise ValueError("Saved prediction checksum mismatch: " + kind)
        predictions[kind] = json.loads(path.read_text(encoding="utf-8"))
    return score_and_save(dataset, predictions, metadata, output)
