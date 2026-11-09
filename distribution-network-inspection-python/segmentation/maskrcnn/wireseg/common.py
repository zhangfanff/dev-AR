import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/wire700.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config["splits"]) != {"train", "val", "test"}:
        raise ValueError("Config must explicitly define train, val and test.")
    if config["model"]["architecture"] != "maskrcnn_resnet50_fpn":
        raise ValueError("This project implements Mask R-CNN with ResNet-50-FPN.")
    validate_config(config)
    return config


def validate_config(config):
    train, evaluation = config["training"], config["evaluation"]
    for key in ("epochs", "batch_size", "accumulation_steps"):
        if train[key] < 1:
            raise ValueError(f"{key} must be positive.")
    if train["workers"] < 0 or train["warmup_steps"] < 0:
        raise ValueError("workers and warmup_steps must be nonnegative.")
    if train["learning_rate"] <= 0 or config["model"]["input_size"] <= 0:
        raise ValueError("Learning rate and input size must be positive.")
    if not 0 <= train["horizontal_flip_probability"] <= 1:
        raise ValueError("Flip probability must lie in [0, 1].")
    if evaluation["max_detections"] != 100:
        raise ValueError(
            "Use max_detections=100 to match the existing COCO comparison."
        )
    if (
        not 0 <= evaluation["score_threshold"] <= 1
        or not 0 < evaluation["mask_threshold"] < 1
    ):
        raise ValueError("Invalid confidence or mask threshold.")


def split_paths(config, split):
    root = Path(config["dataset_root"])
    specification = config["splits"][split]
    return root / specification["annotations"], root / specification["images"]


def dataset_identity(config):
    return {
        split: {
            "annotations": str(split_paths(config, split)[0].resolve()),
            "annotation_sha256": sha256(split_paths(config, split)[0]),
            "images": str(split_paths(config, split)[1].resolve()),
        }
        for split in ("train", "val", "test")
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu.")
    return device


def environment():
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Project checkpoints contain optimizer/RNG state as well as model tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or "model" not in checkpoint:
        raise ValueError("Expected a checkpoint created by this project.")
    return checkpoint


def inference_config(checkpoint, requested_config):
    result = copy.deepcopy(requested_config)
    result["model"] = copy.deepcopy(checkpoint["config"]["model"])
    return result
