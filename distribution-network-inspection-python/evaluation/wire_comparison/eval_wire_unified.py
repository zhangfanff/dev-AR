"""Evaluate RF-DETR or YOLACT predictions with the COCO API."""

import argparse, pathlib, sys, json, hashlib, time, platform
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["yolact", "rfdetr"], required=True)
p.add_argument("--project", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--annotations", default="H:/datasets/LineDatasetFull/test.json")
p.add_argument("--images", default="H:/datasets/LineDatasetFull/spliting_imgs/test")
p.add_argument("--out", required=True)
p.add_argument("--score", type=float, default=0.3)
p.add_argument("--limit", type=int, default=-1)
args = p.parse_args()
sys.path.insert(0, args.project)
if args.model == "yolact":
    from pathlib import Path as _DependencyPath
    import sys as _dependency_sys

    _dependency_sys.path.insert(0, str(_DependencyPath(__file__).resolve().parents[2]))
    from dependencies import activate as _activate_dependency

    _activate_dependency("yolact")
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.manual_seed(42)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this evaluation run")

if args.model == "yolact":
    # Matches the legacy tensor constructor setup in the project's eval.py.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    import cv2
    from data import cfg, set_cfg
    from yolact import Yolact
    from layers.output_utils import postprocess
    from utils.augmentations import BaseTransform

    set_cfg("yolact_new700_test_config")
    cfg.mask_proto_debug = False
    net = Yolact()
    net.load_weights(args.checkpoint)
    net.eval()
    net.cuda()
    net.detect.use_fast_nms = False
    net.detect.use_cross_class_nms = False
    transform = BaseTransform()
    resolution = cfg.max_size

    def infer(path):
        im = cv2.imread(str(path))
        h, w = im.shape[:2]
        x, _, _, _ = transform(
            im,
            np.zeros((1, h, w), np.float32),
            np.array([[0.0, 0.0, 1.0, 1.0]], np.float32),
            {"labels": np.array([0]), "num_crowds": 0},
        )
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
        cls, score, box, masks = postprocess(
            net(x), w, h, crop_masks=True, score_threshold=args.score
        )
        if isinstance(score, list):
            bs, ms = score
        else:
            bs = ms = score
        return (
            cls.cpu().numpy(),
            bs.cpu().numpy(),
            ms.cpu().numpy(),
            box.cpu().numpy(),
            masks.cpu().numpy(),
        )
else:
    import importlib.metadata
    from rfdetr import RFDETRSegPreview

    net = RFDETRSegPreview(pretrain_weights=args.checkpoint)
    resolution = net.model.resolution
    print(
        "RF-DETR resolution",
        resolution,
        "config",
        net.model_config.resolution,
        flush=True,
    )

    def infer(path):
        pred = net.predict(
            np.array(Image.open(path).convert("RGB")), threshold=args.score
        )
        return pred.class_id, pred.confidence, pred.confidence, pred.xyxy, pred.mask


gt = COCO(args.annotations)
gt.dataset.setdefault("info", {})
ids = sorted(gt.getImgIds())
if args.limit > 0:
    ids = ids[: args.limit]
boxes, masks = [], []
start = time.perf_counter()
for it, image_id in enumerate(ids):
    info = gt.imgs[image_id]
    cls, bs, ms, xyxy, segs = infer(pathlib.Path(args.images) / info["file_name"])
    order = np.argsort(-ms)[:100]
    for j in order:
        category_id = int(cls[j])
        if category_id != 0:
            raise ValueError("Expected cable category 0, got " + str(category_id))
        x1, y1, x2, y2 = (float(t) for t in xyxy[j])
        boxes.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(bs[j]),
            }
        )
        rle = mask_utils.encode(np.asfortranarray(segs[j].astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        masks.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "score": float(ms[j]),
            }
        )
    if it % 25 == 0 or it + 1 == len(ids):
        print(
            "PROGRESS",
            args.model,
            it + 1,
            len(ids),
            "predictions",
            len(masks),
            "seconds",
            round(time.perf_counter() - start, 1),
            flush=True,
        )
for kind, preds in [("bbox", boxes), ("segm", masks)]:
    path = out / (args.model + "_" + kind + "_predictions.json")
    path.write_text(json.dumps(preds), encoding="utf-8")
result = {
    "model": args.model,
    "checkpoint": args.checkpoint,
    "checkpoint_sha256": hashlib.sha256(
        pathlib.Path(args.checkpoint).read_bytes()
    ).hexdigest(),
    "annotation_sha256": hashlib.sha256(
        pathlib.Path(args.annotations).read_bytes()
    ).hexdigest(),
    "annotations": args.annotations,
    "image_count": len(ids),
    "gt_instance_count": sum(
        a["image_id"] in set(ids) for a in gt.dataset["annotations"]
    ),
    "score_threshold": args.score,
    "max_detections": 100,
    "resolution": resolution,
    "prediction_count": len(masks),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "metrics": {},
}
(out / (args.model + "_metadata.json")).write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
for kind, preds in [("bbox", boxes), ("segm", masks)]:
    evaluator = COCOeval(gt, gt.loadRes(preds), kind)
    evaluator.params.imgIds = ids
    evaluator.params.catIds = [0]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    result["metrics"][kind] = {
        "AP": float(evaluator.stats[0] * 100),
        "AP50": float(evaluator.stats[1] * 100),
        "AP75": float(evaluator.stats[2] * 100),
        "stats": evaluator.stats.tolist(),
        "AP_by_iou": [
            float(v[v >= 0].mean() * 100)
            for v in evaluator.eval["precision"][:, :, 0, 0, 2]
        ],
    }
path = out / (args.model + "_metrics.json")
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("SAVED", path, flush=True)
