"""Evaluate YOLOv12 with COCO and Ultralytics metrics on the same predictions."""

from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import json
import multiprocessing
import sys

# 在 PyCharm 直接运行时，修改这里选择任务：tower / tower_top。
DEFAULT_TASK = "tower_top"
DEFAULT_SPLIT = "test"
PROJECT = Path(__file__).resolve().parent
WEIGHTS = {
    "tower": PROJECT / "runs/detect/tower/weights/best.pt",
    "tower_top": PROJECT / "runs/detect/train/weights/best.pt",
}
SETTINGS = dict(
    imgsz=640,
    batch=1,
    half=False,
    conf=0.001,
    iou=0.7,
    max_det=100,
    augment=False,
    rect=True,
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=WEIGHTS, default=DEFAULT_TASK)
    parser.add_argument("--split", choices=["test", "val"], default=DEFAULT_SPLIT)
    parser.add_argument(
        "--weights", type=Path, help="Override the task-specific best.pt"
    )
    parser.add_argument(
        "--data-root", type=Path, help="Override a root containing images/ and labels/"
    )
    parser.add_argument("--device", default="0", help="CUDA device, or cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--name", help="New result directory name; existing runs are never overwritten"
    )
    parser.add_argument(
        "--plots", action="store_true", help="Also save native validation plots"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Audit files/labels without inference"
    )
    return parser.parse_args()


def evaluate_task(args):
    # Reuse the exact ground-truth conversion and COCO implementation used in
    # the paper; the new entry point makes the YOLOv12 weight selection explicit.
    sys.path.insert(0, str(PROJECT / "baseline_scripts"))
    import common
    import evaluate as paper_evaluation
    import yaml

    weights = args.weights or WEIGHTS[args.task]
    weights = (weights if weights.is_absolute() else PROJECT / weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if args.data_root:
        data_root = (
            args.data_root if args.data_root.is_absolute() else PROJECT / args.data_root
        )
        common.TASKS[args.task] = dict(
            common.TASKS[args.task], root=data_root.resolve()
        )
    summary, _ = common.audit(args.task, args.split)
    if not summary["class_instances"][0]:
        raise ValueError("The selected split contains no class-0 targets to evaluate")
    dataset = common.TASKS[args.task]
    print(
        json.dumps(
            {
                "task": args.task,
                "split": args.split,
                "weights": str(weights),
                "dataset_root": str(dataset["root"]),
                "dataset": summary,
                "settings": SETTINGS,
                "target_class": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN OK: no inference and no evaluation results written.")
        return

    import torch
    from dependencies import activate

    activate("yolov12")
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect.val import DetectionValidator

    model = YOLO(str(weights))
    # Validation can fuse Conv/BN in place; count the checkpoint before fusion,
    # matching the parameter totals used in the paper.
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    expected = dict(enumerate(dataset["names"]))
    if model.names != expected:
        raise ValueError(
            f"Wrong task weights: model classes={model.names}; expected={expected}"
        )
    # This describes what model.val(split=...) would inherit in this checkout;
    # the actual evaluation below deliberately supplies every paper setting.
    inherited = get_cfg(
        overrides={**model.overrides, "rect": True, "split": args.split, "mode": "val"}
    )
    keys = ["data", "split", "imgsz", "batch", "conf", "iou", "max_det", "half", "rect"]
    inherited_settings = {key: getattr(inherited, key) for key in keys}
    if inherited_settings["conf"] is None:
        inherited_settings["conf"] = 0.001  # BaseValidator's validation fallback

    name = args.name or f"{args.task}_{args.split}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("--name must be a directory name, without parent directories")
    folder = PROJECT / "runs/yolo12_eval" / name
    if folder.exists():
        raise FileExistsError(f"Choose a new --name: {folder}")
    folder.mkdir(parents=True)
    gt, image_ids = paper_evaluation.ground_truth(args.task, args.split)
    data_file = folder / "data.yaml"
    data_file.write_text(
        yaml.safe_dump(
            {
                "path": dataset["root"].as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(expected),
                "names": dataset["names"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    save_json(folder / "ground_truth_coco.json", gt)
    holder = {}

    class ExportValidator(DetectionValidator):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            holder["validator"] = self

        def pred_to_json(self, predn, filename):
            # Keep full precision and the original class IDs. Both evaluators
            # consume the same predictions, rather than rerunning inference.
            image_id = image_ids[str(Path(filename).resolve())]
            for x1, y1, x2, y2, score, cls in predn.cpu().tolist():
                self.jdict.append(
                    {
                        "image_id": image_id,
                        "category_id": int(cls),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )

    manifest = {
        "model": "yolov12n",
        "task": args.task,
        "split": args.split,
        "weights": str(weights),
        "weight_sha256": digest(weights),
        "dataset_root": str(dataset["root"]),
        "data": str(data_file),
        "images": summary["images"],
        "instances": summary["instances"],
        "class_instances": summary["class_instances"],
        "label_and_filename_sha256": summary["label_and_filename_sha256"],
        "settings": SETTINGS,
        "pr_conf": 0.25,
        "pr_iou": 0.5,
        "original_val_call_inherited_settings": inherited_settings,
        "evaluator_sha256": digest(PROJECT / "baseline_scripts/evaluate.py"),
        "torch_version": torch.__version__,
    }
    save_json(folder / "evaluation_manifest.json", manifest)
    metrics = model.val(
        data=str(data_file),
        split=args.split,
        validator=ExportValidator,
        **SETTINGS,
        device=args.device,
        workers=args.workers,
        plots=args.plots,
        save_json=True,
        project=str(folder.parent),
        name=folder.name,
        exist_ok=True,
    )
    validator = holder["validator"]
    if validator.seen != summary["images"]:
        raise RuntimeError(
            f"Image-count mismatch: expected {summary['images']}, saw {validator.seen}"
        )
    predictions = validator.jdict
    save_json(folder / "predictions_coco.json", predictions)
    classes, average = paper_evaluation.coco_metrics(gt, predictions)

    native_classes = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        native_classes[int(class_id)] = {
            "name": expected[int(class_id)],
            "P_percent": float(metrics.box.p[index] * 100),
            "R_percent": float(metrics.box.r[index] * 100),
            "AP50_percent": float(metrics.box.ap50[index] * 100),
            "AP_percent": float(metrics.box.ap[index] * 100),
        }
    native = {
        "class_metrics": native_classes,
        "all_classes_mean_AP": {
            "AP50_percent": float(metrics.box.map50 * 100),
            "AP_percent": float(metrics.box.map * 100),
        },
        "P_R_definition": "Ultralytics P/R at the smoothed mean-F1 optimum",
    }
    if metrics.box.f1_curve.size:
        from ultralytics.utils.metrics import smooth

        best = smooth(metrics.box.f1_curve.mean(0), 0.1).argmax()
        native["max_f1_confidence"] = float(metrics.box.px[best])
    actual = {key: getattr(validator.args, key) for key in keys}
    report = {
        **manifest,
        "parameters": parameters,
        "parameters_after_validation_fusion": sum(
            p.numel() for p in model.model.parameters()
        ),
        "parameter_definition": "Checkpoint parameters before validation Conv/BN fusion",
        "actual_settings": actual,
        "class_metrics": classes,
        "primary_target_metrics": classes[0],
        "all_classes_mean_AP": average,
        "native_ultralytics": native,
        "speed_ms": metrics.speed,
        "latency_ms": sum(
            metrics.speed[k] for k in ["preprocess", "inference", "postprocess"]
        ),
        "metric_definition": "Paper: pycocotools COCO bbox AP; P/R at conf>=0.25, IoU=0.50; maxDet=100",
        "latency_definition": "Batch=1 FP32 validator profiling; preprocessing+inference+postprocessing; excludes model loading and disk IO",
    }
    save_json(folder / "metrics.json", report)
    with (folder / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "implementation",
                "class_id",
                "class",
                "P_percent",
                "R_percent",
                "AP50_percent",
                "AP_percent",
            ]
        )
        for implementation, rows in [
            ("COCO_fixed_conf_0.25", classes),
            ("Ultralytics_max_F1", native_classes),
        ]:
            for class_id, values in rows.items():
                writer.writerow(
                    [implementation, class_id, expected[class_id]]
                    + [
                        values[k]
                        for k in [
                            "P_percent",
                            "R_percent",
                            "AP50_percent",
                            "AP_percent",
                        ]
                    ]
                )
    print(
        "\nSame predictions, different evaluation implementations (all values are %):"
    )
    print(f"{'Method / scope':34s} {'P':>8} {'R':>8} {'AP50':>8} {'AP50-95':>9}")
    for label, values in [
        ("Paper COCO / target class 0", classes[0]),
        ("Ultralytics / target class 0", native_classes[0]),
    ]:
        print(
            f"{label:34s} {values['P_percent']:8.2f} {values['R_percent']:8.2f} "
            f"{values['AP50_percent']:8.2f} {values['AP_percent']:9.2f}"
        )
    for label, values in [
        ("COCO / all-class mean", average),
        ("Ultralytics / all-class mean", native["all_classes_mean_AP"]),
    ]:
        print(
            f"{label:34s} {'--':>8} {'--':>8} {values['AP50_percent']:8.2f} {values['AP_percent']:9.2f}"
        )
    print(
        f"\nPaper table uses primary_target_metrics (class 0); all-class means are listed separately. Results: {folder}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    multiprocessing.freeze_support()
    evaluate_task(parse_args())
