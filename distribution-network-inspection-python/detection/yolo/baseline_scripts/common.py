from pathlib import Path as _DependencyPath
import sys as _dependency_sys

_dependency_sys.path.insert(0, str(_DependencyPath(__file__).resolve().parents[3]))
from dependencies import activate as _activate_dependency

_activate_dependency("yolov12", required=False)

"""Dataset definitions and shared settings for tower detection tasks."""

from pathlib import Path
import os
import argparse
import json
import hashlib
from datetime import datetime
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
TASKS = {
    "tower": {
        "root": Path(os.environ.get("TOWER_DATASET_ROOT", "datasets/tower")),
        "names": ["tower"],
    },
    "tower_top": {
        "root": Path(os.environ.get("TOWER_TOP_DATASET_ROOT", "datasets/tower_top")),
        "names": ["tower-top", "hengdan", "jinggai", "car", "byq"],
    },
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parser(model, mode):
    p = argparse.ArgumentParser(
        description=f"{model}: {mode}; independent tower/tower_top dataset configurations"
    )
    p.add_argument("--task", choices=TASKS, default="tower")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--weights",
        help="Pretrained weights for training; trained best.pt for validation",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Check data and print configuration without training or inference",
    )
    if mode == "train":
        p.add_argument("--epochs", type=int, default=100)
        p.add_argument("--batch", type=int, help="Default: tower=32, tower_top=16")
        p.add_argument("--name", help="Default: paper_<task>_<model>")
    else:
        p.add_argument("--split", choices=["val", "test"], default="val")
        p.add_argument("--name", help="Evaluation run name")
    return p


def run_name(task, model):
    return f"paper_{task}_{model}"


def data_config(task):
    info = TASKS[task]
    config = SCRIPTS / "configs" / f"{task}.yaml"
    config.parent.mkdir(exist_ok=True)
    # Never inherit the root data.yaml: it points to a different task.
    content = {
        "path": info["root"].as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(info["names"]),
        "names": info["names"],
    }
    config.write_text(
        yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config


def audit(task, split):
    info = TASKS[task]
    root = info["root"]
    images = root / "images" / split
    labels = root / "labels" / split
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"Missing images/labels: {root}, {split}")
    files = sorted(
        p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    if not files:
        raise ValueError(f"No images in {images}")
    counts = [0] * len(info["names"])
    empty = 0
    digest = hashlib.sha256()
    for image in files:
        relative = image.relative_to(images)
        label = labels / relative.with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(label)
        raw = label.read_bytes()
        digest.update(relative.as_posix().encode())
        digest.update(raw)
        rows = [
            line.split()
            for line in raw.decode("utf-8-sig").splitlines()
            if line.strip()
        ]
        empty += not bool(rows)
        for row in rows:
            if len(row) != 5:
                raise ValueError(f"Not a YOLO detection label: {label}: {row}")
            cls, cx, cy, w, h = map(float, row)
            if (
                int(cls) != cls
                or not 0 <= cls < len(counts)
                or not all(0 <= v <= 1 for v in [cx, cy, w, h])
                or w <= 0
                or h <= 0
            ):
                raise ValueError(f"Invalid bounding box: {label}: {row}")
            counts[int(cls)] += 1
    return {
        "images": len(files),
        "instances": sum(counts),
        "class_instances": counts,
        "empty_images": empty,
        "label_and_filename_sha256": digest.hexdigest(),
    }, files


def train_settings(task):
    top = task == "tower_top"
    return dict(
        epochs=100,
        batch=16 if top else 32,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        seed=0,
        amp=True,
        nbs=64,
        cos_lr=False,
        patience=100,
        cache=False,
        degrees=60.0 if top else 0.0,
        translate=0.3 if top else 0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.5 if top else 0.0,
        fliplr=0.5,
        bgr=0.5 if top else 0.0,
        mosaic=0.5 if top else 1.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        close_mosaic=10,
    )


def preparation(args, model, mode):
    splits = ["train", "val"] if mode == "train" else [args.split]
    summaries = {split: audit(args.task, split)[0] for split in splits}
    config = data_config(args.task)
    settings = (
        train_settings(args.task)
        if mode == "train"
        else dict(
            imgsz=640,
            batch=1,
            half=False,
            conf=0.001,
            iou=0.7,
            max_det=100,
            pr_conf=0.25,
            pr_iou=0.5,
        )
    )
    if mode == "train":
        settings["epochs"] = args.epochs
        if args.batch is not None:
            settings["batch"] = args.batch
        if settings["epochs"] <= 0 or settings["batch"] <= 0:
            raise ValueError("epochs and batch must be positive")
    weights = (
        Path(args.weights).resolve()
        if args.weights
        else (
            PROJECT / f"{model}.pt"
            if mode == "train"
            else PROJECT
            / "runs/detect"
            / run_name(args.task, model)
            / "weights/best.pt"
        )
    )
    report = {
        "model": model,
        "task": args.task,
        "mode": mode,
        "data": str(config),
        "dataset_root": str(TASKS[args.task]["root"]),
        "weights": str(weights),
        "settings": settings,
        "partitions": summaries,
        "device": args.device,
        "workers": args.workers,
        "name": args.name or run_name(args.task, model),
        "created_at": datetime.now().isoformat(),
    }
    if (
        mode == "train"
        and not args.dry_run
        and (PROJECT / "runs/detect" / report["name"]).exists()
    ):
        raise FileExistsError(
            f"Training run already exists; choose a new --name: {report['name']}"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        folder = PROJECT / "runs/baseline_manifests"
        folder.mkdir(parents=True, exist_ok=True)
        (
            folder / f"{report['name']}_{mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
        ).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
