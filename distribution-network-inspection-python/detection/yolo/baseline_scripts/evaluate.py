"""Shared COCO metrics; native framework CSV AP/P/R are not test metrics."""

from pathlib import Path
import json
import sys
import hashlib
import numpy as np
from PIL import Image
from common import PROJECT, TASKS, audit, preparation, parser


def ground_truth(task, split):
    info = TASKS[task]
    _, files = audit(task, split)
    gt = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(info["names"])],
    }
    ids = {}
    for iid, p in enumerate(files, 1):
        ids[str(p.resolve())] = iid
        with Image.open(p) as im:
            w, h = im.size
        gt["images"].append({"id": iid, "file_name": p.name, "width": w, "height": h})
        label = (
            info["root"]
            / "labels"
            / split
            / p.relative_to(info["root"] / "images" / split).with_suffix(".txt")
        )
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = map(float, line.split())
            gt["annotations"].append(
                {
                    "id": len(gt["annotations"]) + 1,
                    "image_id": iid,
                    "category_id": int(cls),
                    "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                    "area": bw * w * bh * h,
                    "iscrowd": 0,
                }
            )
    return gt, ids


def run_v8(report, args, ids, folder):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dependencies import activate

    activate("yolov12")
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator

    holder = {}

    class Export(DetectionValidator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            holder["v"] = self

        def pred_to_json(self, predn, filename):
            for x1, y1, x2, y2, score, cls in predn.cpu().tolist():
                self.jdict.append(
                    {
                        "image_id": ids[str(Path(filename).resolve())],
                        "category_id": int(cls),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )

    model = YOLO(report["weights"])
    parameters = sum(p.numel() for p in model.model.parameters())
    expected = {i: n for i, n in enumerate(TASKS[args.task]["names"])}
    if model.names != expected:
        raise ValueError(f"Wrong task checkpoint: {model.names}; expected {expected}")
    metrics = model.val(
        data=report["data"],
        split=args.split,
        validator=Export,
        imgsz=640,
        batch=1,
        half=False,
        conf=0.001,
        iou=0.7,
        max_det=100,
        device=args.device,
        workers=args.workers,
        augment=False,
        plots=False,
        save_json=True,
        project=str(folder.parent),
        name=folder.name,
        exist_ok=True,
    )
    assert holder["v"].seen == len(ids)
    return holder["v"].jdict, metrics.speed, parameters


def run_v5(report, args, ids, folder):
    from v5_runtime import activate

    activate()
    import torch
    import val

    checkpoint = torch.load(report["weights"], map_location="cpu")
    model = checkpoint.get("ema") or checkpoint["model"]
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    expected = dict(enumerate(TASKS[args.task]["names"]))
    if names != expected:
        raise ValueError(f"Wrong task checkpoint: {names}; expected {expected}")
    parameters = sum(p.numel() for p in model.parameters())
    del checkpoint, model

    def export(predn, jdict, path, class_map):
        for x1, y1, x2, y2, score, cls in predn.cpu().tolist():
            jdict.append(
                {
                    "image_id": ids[str(Path(path).resolve())],
                    "category_id": int(cls),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )

    val.save_one_json = export
    results, maps, times = val.run(
        data=report["data"],
        weights=report["weights"],
        task=args.split,
        imgsz=640,
        batch_size=1,
        half=False,
        conf_thres=0.001,
        iou_thres=0.7,
        max_det=100,
        device=args.device,
        workers=args.workers,
        augment=False,
        plots=False,
        save_json=True,
        project=str(folder.parent),
        name=folder.name,
        exist_ok=True,
    )
    pred_file = folder / (Path(report["weights"]).stem + "_predictions.json")
    preds = json.loads(pred_file.read_text()) if pred_file.exists() else []
    return (
        preds,
        dict(zip(["preprocess", "inference", "postprocess"], map(float, times))),
        parameters,
    )


def coco_metrics(gt, preds):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    api = COCO()
    api.dataset = gt
    api.createIndex()
    if preds:
        dt = api.loadRes(preds)
    else:
        dt = COCO()
        dt.dataset = {**gt, "annotations": []}
        dt.createIndex()
    ev = COCOeval(api, dt, "bbox")
    ev.params.imgIds = [i["id"] for i in gt["images"]]
    ev.params.catIds = [c["id"] for c in gt["categories"]]
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    output = {}
    for k, cat in enumerate(ev.params.catIds):
        cat = int(cat)
        precision = ev.eval["precision"][:, :, k, 0, -1]
        valid = precision[precision >= 0]
        valid50 = precision[0][precision[0] >= 0]
        tp = fp = 0
        for r in ev.evalImgs:
            if (
                r is None
                or r["category_id"] != cat
                or list(r["aRng"]) != list(ev.params.areaRng[0])
            ):
                continue
            keep = (np.asarray(r["dtScores"]) >= 0.25) & ~r["dtIgnore"][0]
            tp += int(np.sum(keep & (r["dtMatches"][0] > 0)))
            fp += int(np.sum(keep & (r["dtMatches"][0] == 0)))
        targets = sum(a["category_id"] == cat for a in gt["annotations"])
        fn = targets - tp
        output[cat] = {
            "P_percent": 100 * tp / (tp + fp) if tp + fp else 0.0,
            "R_percent": 100 * tp / targets if targets else None,
            "AP50_percent": 100 * float(valid50.mean()) if valid50.size else None,
            "AP_percent": 100 * float(valid.mean()) if valid.size else None,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }
    return output, {
        "AP_percent": float(ev.stats[0] * 100),
        "AP50_percent": float(ev.stats[1] * 100),
    }


def main(model):
    args = parser(model, "validate").parse_args()
    report = preparation(args, model, "validate")
    if args.dry_run:
        return
    weight = Path(report["weights"])
    if not weight.is_file():
        raise FileNotFoundError(f"Train first or specify --weights: {weight}")
    folder = (
        PROJECT
        / "runs/baseline_eval"
        / (args.name or f"{args.task}_{model}_{args.split}")
    )
    if folder.exists():
        raise FileExistsError(
            f"Evaluation directory exists; choose a new --name: {folder}"
        )
    folder.mkdir(parents=True)
    gt, ids = ground_truth(args.task, args.split)
    preds, speed, parameters = (run_v8 if model == "yolov8n" else run_v5)(
        report, args, ids, folder
    )
    classes, average = coco_metrics(gt, preds)
    result = {
        **report,
        "weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        "parameters": parameters,
        "class_metrics": classes,
        "primary_target_metrics": classes[0],
        "all_classes_mean_AP": average,
        "speed_ms": speed,
        "latency_ms": sum(speed[k] for k in ["preprocess", "inference", "postprocess"]),
        "metric_definition": "pycocotools COCO AP; P/R at confidence >=0.25, IoU 0.50; maxDet=100",
        "latency_definition": "batch=1 FP32; synchronized validator profiling after model warmup; excludes disk IO and model loading",
    }
    for filename, obj in [
        ("metrics.json", result),
        ("predictions_coco.json", preds),
        ("ground_truth_coco.json", gt),
    ]:
        (folder / filename).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
