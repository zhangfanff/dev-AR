"""ONNX test-set AP vs resolution. Desktop throughput is not controller throughput."""

from pathlib import Path
import argparse, copy, csv, json, sys, time
import numpy as np
from PIL import Image
from prepare_models import ROOT, YOLO, RF, sha


def preprocess(image, config):
    w, h = image.size
    s = config["width"]
    wire = config["task"] == "wire"
    if wire:
        image = image.resize((s, s), Image.Resampling.BILINEAR)
        scale = None
        left = top = 0
    else:
        scale = min(s / w, s / h)
        rw, rh = round(w * scale), round(h * scale)
        left, top = round((s - rw) / 2 - 0.1), round((s - rh) / 2 - 0.1)
        canvas = Image.new("RGB", (s, s), (114, 114, 114))
        canvas.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (left, top))
        image = canvas
    x = np.asarray(image, dtype=np.float32) / 255
    if wire:
        x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), (scale, left, top)


def nms(boxes, scores, classes, threshold=0.7, limit=100):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order) and len(keep) < limit:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if not len(rest):
            break
        lo = np.maximum(boxes[i, :2], boxes[rest, :2])
        hi = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.maximum(hi - lo, 0).prod(1)
        area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-10)
        order = rest[(classes[rest] != classes[i]) | (iou <= threshold)]
    return keep


def decode_yolo(raw, config, geometry, w, h, confidence=0.001):
    a = raw[0].T
    cls = a[:, 4:].argmax(1)
    scores = a[np.arange(len(a)), 4 + cls]
    keep = scores >= confidence
    a = a[keep]
    cls = cls[keep]
    scores = scores[keep]
    boxes = np.concatenate([a[:, :2] - a[:, 2:4] / 2, a[:, :2] + a[:, 2:4] / 2], axis=1)
    ids = nms(boxes, scores, cls, config.get("iou", 0.7))
    boxes = boxes[ids]
    scores = scores[ids]
    cls = cls[ids]
    scale, left, top = geometry
    boxes = (boxes - np.array([left, top, left, top])) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes, scores, cls


def wire_predictions(outputs, config, iid, w, h, category):
    import torch
    import torch.nn.functional as F
    from pycocotools import mask as mask_api

    boxes, logits, masks = [o[0] for o in outputs]
    assert logits.shape[1] == 1
    scores = 1 / (1 + np.exp(-np.clip(logits[:, 0], -80, 80)))
    ids = np.argsort(-scores, kind="stable")
    ids = ids[scores[ids] >= 0.3][:100]
    pred = []
    if not len(ids):
        return pred
    scaled = (
        F.interpolate(
            torch.from_numpy(masks[ids, None]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()
        > 0
    )
    for q, mask in zip(ids, scaled):
        rle = mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        pred.append(
            dict(
                image_id=iid,
                category_id=category,
                score=float(scores[q]),
                segmentation=rle,
            )
        )
    return pred


def dataset(task):
    if task == "wire":
        folder = RF / "dataset_new/test"
        gt = json.loads((folder / "_annotations.coco.json").read_text(encoding="utf-8"))
        gt.setdefault("info", {})
        return gt, {i["id"]: folder / i["file_name"] for i in gt["images"]}
    # Reuse the frozen, already evaluated test-set COCO annotations and filenames.
    folder = ROOT / "reference_annotations"
    source = folder / f"{task}_ground_truth_coco.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    gt = json.loads(source.read_text(encoding="utf-8"))
    gt.setdefault("info", {})
    image_root = (
        Path(r"I:\dataset_tower\peiwang_no5910_v4\images\test")
        if task == "tower"
        else Path(
            r"I:\datasets\tower-top-addlightimg-addlightimg-4class-addshiqu-add60-addblanktowertop-adddouble\images\test"
        )
    )
    paths = {}
    for i in gt["images"]:
        p = image_root / i["file_name"]
        if not p.is_file():
            p = image_root / Path(i["file_name"]).name
        if not p.is_file():
            raise FileNotFoundError(p)
        paths[i["id"]] = p
    return gt, paths


def metrics(gt, preds, kind):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    api = COCO()
    api.dataset = gt
    api.createIndex()
    if preds:
        dt = api.loadRes(copy.deepcopy(preds))
    else:
        dt = COCO()
        dt.dataset = {**gt, "annotations": []}
        dt.createIndex()
    e = COCOeval(api, dt, kind)
    e.params.maxDets = [1, 10, 100]
    e.evaluate()
    e.accumulate()
    e.summarize()
    target = next(
        i for i, c in enumerate(e.params.catIds) if c == gt["categories"][0]["id"]
    )
    a = e.eval["precision"][:, :, target, 0, -1]
    a50 = a[0]
    return dict(
        AP=float(a[a >= 0].mean() * 100),
        AP50=float(a50[a50 >= 0].mean() * 100),
        all_class_AP=float(e.stats[0] * 100),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=ROOT / "models")
    p.add_argument("--out", type=Path, default=ROOT / "resolution_results")
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["tower", "tower_top", "wire"],
        default=["tower", "tower_top", "wire"],
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke test only; nonzero results are not full test-set metrics",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort

    results = []
    for config_path in sorted(args.models.glob("*/model.json")):
        config = json.loads(config_path.read_text())
        task = config["task"]
        if task not in args.tasks:
            continue
        model = config_path.parent / config["file"]
        assert sha(model) == config["onnx_sha256"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(model), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        gt, paths = dataset(task)
        images = gt["images"][: args.limit] if args.limit else gt["images"]
        if args.limit:
            ids = {i["id"] for i in images}
            gt = {
                **gt,
                "images": images,
                "annotations": [a for a in gt["annotations"] if a["image_id"] in ids],
            }
        preds = []
        for index, i in enumerate(images):
            with Image.open(paths[i["id"]]) as im:
                image = im.convert("RGB")
            w, h = image.size
            x, geo = preprocess(image, config)
            outputs = session.run(config["outputs"], {config["input"]: x})
            if task == "wire":
                preds.extend(
                    wire_predictions(
                        outputs, config, i["id"], w, h, gt["categories"][0]["id"]
                    )
                )
            else:
                boxes, scores, classes = decode_yolo(outputs[0], config, geo, w, h)
                for b, s, c in zip(boxes, scores, classes):
                    preds.append(
                        dict(
                            image_id=i["id"],
                            category_id=int(c),
                            bbox=[
                                float(b[0]),
                                float(b[1]),
                                float(b[2] - b[0]),
                                float(b[3] - b[1]),
                            ],
                            score=float(s),
                        )
                    )
            if (index + 1) % 20 == 0:
                print(config_path.parent.name, index + 1, "/", len(images), flush=True)
        kind = "segm" if task == "wire" else "bbox"
        row = dict(
            model=config_path.parent.name,
            task=task,
            resolution=config["width"],
            images=len(images),
            smoke_test=bool(args.limit),
            onnx_sha256=config["onnx_sha256"],
            **metrics(gt, preds, kind),
        )
        results.append(row)
        (args.out / (config_path.parent.name + "_predictions.json")).write_text(
            json.dumps(preds), encoding="utf-8"
        )
        (args.out / "accuracy.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    if results:
        with (args.out / "accuracy.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)


if __name__ == "__main__":
    main()
