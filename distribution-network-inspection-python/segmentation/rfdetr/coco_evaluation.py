"""Complete-image COCO box/mask evaluation shared by the two inspection projects."""

import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROTOCOL_VERSION = 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value, root):
    path = Path(value)
    return path if path.is_absolute() else root / path


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


def _predictor(model_kind, args, cfg, checkpoint):
    import torch

    device = "cuda" if args.cuda else "cpu"
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --cuda=false to evaluate on CPU.")
    seed = args.seed if args.seed is not None else 42
    torch.manual_seed(seed)
    if model_kind == "yolact":
        import cv2
        from yolact import Yolact
        from layers.output_utils import postprocess
        from utils.augmentations import BaseTransform

        # Required by the project's legacy torch.Tensor(..., device=...) constructors.
        torch.set_default_tensor_type(
            "torch.cuda.FloatTensor" if args.cuda else "torch.FloatTensor"
        )
        cfg.mask_proto_debug = False
        cfg.max_num_detections = args.max_detections
        net = Yolact()
        net.load_weights(str(checkpoint))
        net.eval()
        net.to(device)
        net.detect.use_fast_nms = args.fast_nms
        net.detect.use_cross_class_nms = args.cross_class_nms
        transform = BaseTransform()

        def predict(path):
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError("Cannot read image: " + str(path))
            h, w = image.shape[:2]
            # Transform only image pixels. Actual GT never enters Resize's box filter.
            image, _, _, _ = transform(
                image,
                np.zeros((1, h, w), np.float32),
                np.array([[0.0, 0.0, 1.0, 1.0]], np.float32),
                {"labels": np.array([0]), "num_crowds": 0},
            )
            batch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                classes, scores, boxes, masks = postprocess(
                    net(batch),
                    w,
                    h,
                    crop_masks=args.crop,
                    score_threshold=args.score_threshold,
                )
            box_scores, mask_scores = (
                scores if isinstance(scores, list) else (scores, scores)
            )
            return (
                classes.cpu().numpy(),
                box_scores.cpu().numpy(),
                mask_scores.cpu().numpy(),
                boxes.cpu().numpy(),
                masks.cpu().numpy(),
            )

        return predict, {
            "resolution": cfg.max_size,
            "fast_nms": args.fast_nms,
            "nms_iou": cfg.nms_thresh,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    from PIL import Image
    from rfdetr import RFDETRSegPreview
    import importlib.metadata

    model = RFDETRSegPreview(pretrain_weights=str(checkpoint), device=device)

    def predict(path):
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"))
        with torch.no_grad():
            result = model.predict(pixels, threshold=args.score_threshold)
        return (
            result.class_id,
            result.confidence,
            result.confidence,
            result.xyxy,
            result.mask,
        )

    return predict, {
        "resolution": model.model.resolution,
        "fast_nms": None,
        "nms_iou": None,
        "rfdetr": importlib.metadata.version("rfdetr"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def run_standard_evaluation(model_kind, args, cfg, project_root):
    root = Path(project_root).resolve()
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("--score_threshold must be between 0 and 1.")
    if args.max_detections < 10:
        raise ValueError("--max_detections must be at least 10 for COCO summaries.")
    if args.max_images == 0 or args.max_images < -1:
        raise ValueError("--max_images must be -1 or a positive integer.")
    if model_kind == "yolact" and args.score_threshold < cfg.nms_conf_thresh:
        raise ValueError(
            "Requested threshold is below YOLACT nms_conf_thresh; adjust the config explicitly."
        )
    annotation = _path(cfg.dataset.valid_info, root)
    images = _path(cfg.dataset.valid_images, root)
    checkpoint = _path(args.trained_model, root)
    output = _path(args.eval_output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    gt = COCO(str(annotation))
    image_ids = sorted(gt.getImgIds())  # Includes unannotated/negative images.
    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]
    selected = set(image_ids)
    if not selected:
        raise ValueError("No images to evaluate.")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "model": model_kind,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "annotations": str(annotation),
        "annotation_sha256": _sha256(annotation),
        "image_ids": image_ids,
        "image_count": len(image_ids),
        "gt_instance_count": sum(
            a["image_id"] in selected for a in gt.dataset["annotations"]
        ),
        "empty_gt_images": sum(not gt.getAnnIds(imgIds=[i]) for i in image_ids),
        "score_threshold": args.score_threshold,
        "max_detections": args.max_detections,
        "crop_masks": args.crop if model_kind == "yolact" else None,
        "fast_nms": args.fast_nms if model_kind == "yolact" else None,
        "cross_class_nms": args.cross_class_nms if model_kind == "yolact" else None,
        "detect_only": args.detect,
    }
    paths = {
        kind: output / (model_kind + "_" + kind + "_predictions.json")
        for kind in ("bbox", "segm")
    }
    meta_path = output / (model_kind + "_metadata.json")
    if args.reuse_predictions or args.resume:
        if not meta_path.exists():
            raise ValueError(
                "No standard COCO prediction metadata found. Old ap_data.pkl cannot reconstruct "
                "omitted GT. Run inference once without --resume/--reuse_predictions."
            )
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if key in ("checkpoint", "annotations") and metadata.get(key) is not None:
                if Path(metadata[key]).resolve() == Path(value).resolve():
                    continue
            if metadata.get(key) != value:
                raise ValueError("Saved predictions have a different protocol: " + key)
        predictions = {
            kind: json.loads(path.read_text(encoding="utf-8"))
            for kind, path in paths.items()
            if kind != "segm" or not args.detect
        }
        for kind, rows in predictions.items():
            if _sha256(paths[kind]) != metadata["prediction_sha256"][kind]:
                raise ValueError("Saved prediction checksum mismatch: " + kind)
    else:
        predictor, model_metadata = _predictor(model_kind, args, cfg, checkpoint)
        metadata = dict(expected, **model_metadata)
        if args.mask_proto_debug:
            print(
                "Note: --mask_proto_debug is ignored in quantitative COCO evaluation."
            )
        label_map = cfg.dataset.label_map
        class_to_category = (
            {int(v) - 1: int(k) for k, v in label_map.items()}
            if label_map is not None
            else {i: i + 1 for i in range(len(cfg.dataset.class_names))}
        )
        predictions = {"bbox": [], "segm": []}
        started = time.perf_counter()
        for index, image_id in enumerate(image_ids):
            info = gt.imgs[image_id]
            classes, box_scores, mask_scores, boxes, masks = predictor(
                images / info["file_name"]
            )
            ranking = box_scores if args.detect else mask_scores
            for j in np.argsort(-ranking, kind="stable")[: args.max_detections]:
                category_id = class_to_category[int(classes[j])]
                if category_id not in gt.cats:
                    raise ValueError(
                        "Prediction category absent from GT: " + str(category_id)
                    )
                x1, y1, x2, y2 = [float(v) for v in boxes[j]]
                predictions["bbox"].append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(box_scores[j]),
                    }
                )
                if not args.detect:
                    mask = np.asarray(masks[j])
                    if mask.shape != (info["height"], info["width"]):
                        raise ValueError("Mask shape mismatch for " + info["file_name"])
                    encoded = mask_utils.encode(
                        np.asfortranarray((mask > 0).astype(np.uint8))
                    )
                    encoded["counts"] = encoded["counts"].decode("ascii")
                    predictions["segm"].append(
                        {
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": encoded,
                            "score": float(mask_scores[j]),
                        }
                    )
            if not args.no_bar and (index % 25 == 0 or index + 1 == len(image_ids)):
                print(
                    "COCO inference %d/%d (%.1fs)"
                    % (index + 1, len(image_ids), time.perf_counter() - started),
                    flush=True,
                )
        metadata["python"] = platform.python_version()
        metadata["prediction_count"] = len(predictions["bbox"])
        metadata["prediction_sha256"] = {}
        for kind, rows in predictions.items():
            paths[kind].write_text(json.dumps(rows), encoding="utf-8")
            metadata["prediction_sha256"][kind] = _sha256(paths[kind])
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        "Evaluation population: %d images, %d GT instances, %d negative images; threshold %.3f"
        % (
            expected["image_count"],
            expected["gt_instance_count"],
            expected["empty_gt_images"],
            args.score_threshold,
        )
    )
    metrics = evaluate_predictions(
        gt, predictions, image_ids, args.max_detections, args.detect
    )
    print_per_class(metrics)
    report = dict(metadata, metrics=metrics)
    destination = output / (model_kind + "_metrics.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved metrics:", destination)
    return report
