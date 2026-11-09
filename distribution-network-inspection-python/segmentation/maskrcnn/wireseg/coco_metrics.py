"""Unmodified COCO metric functions from RF-DETR/utils/coco_evaluation.py.
Same full-population metric kernel as YOLACT and RF-DETR.
"""

import copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_predictions(
    gt, predictions, image_ids, max_detections=100, detect_only=False
):
    """Retain every original GT annotation, including frames with zero predictions."""
    gt.dataset.setdefault("info", {})
    metrics = {}
    for kind in ("bbox",) if detect_only else ("bbox", "segm"):
        rows = predictions[kind]
        if rows:
            result = gt.loadRes(copy.deepcopy(rows))
        else:
            # COCO.loadRes([]) indexes the first element; construct an empty result safely.
            result = COCO()
            result.dataset = {
                "images": copy.deepcopy(gt.dataset["images"]),
                "categories": copy.deepcopy(gt.dataset["categories"]),
                "annotations": [],
            }
            result.createIndex()
        evaluator = COCOeval(gt, result, kind)
        evaluator.params.imgIds = list(image_ids)
        evaluator.params.catIds = sorted(gt.getCatIds())
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        precision = evaluator.eval["precision"][:, :, :, 0, -1]

        def average(values):
            values = values[values >= 0]
            return float(values.mean() * 100) if values.size else -1.0

        per_class = []
        for index, category_id in enumerate(evaluator.params.catIds):
            category_id = int(category_id)
            per_class.append(
                {
                    "category_id": category_id,
                    "name": gt.cats[category_id]["name"],
                    "AP": average(precision[:, :, index]),
                    "AP_by_iou": [average(p[:, index]) for p in precision],
                }
            )
        metrics[kind] = {
            "AP": average(precision),
            "AP50": average(precision[0]),
            "AP75": average(precision[5]),
            "AP_by_iou": [average(p) for p in precision],
            "per_class": per_class,
            "stats": evaluator.stats.tolist(),
        }
    return metrics


def print_per_class(metrics):
    headers = ["class/type", "all"] + [".%02d" % n for n in range(50, 100, 5)]
    print("\nPer-Class AP Details (percent; standard COCO, IoU 0.50:0.95)")
    print(" | ".join("%12s" % value for value in headers))
    for kind, values in metrics.items():
        for row in values["per_class"]:
            name = row["name"] + ("(box)" if kind == "bbox" else "(mask)")
            # AP is calculated directly from the ten thresholds, never from a missing key.
            print(
                " | ".join(
                    "%12s" % value
                    for value in [name]
                    + ["%.2f" % n for n in [row["AP"]] + row["AP_by_iou"]]
                )
            )
