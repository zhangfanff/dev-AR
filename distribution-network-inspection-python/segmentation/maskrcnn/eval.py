"""Evaluate validation or test data; train.py selects checkpoints on validation only."""

import argparse
from pathlib import Path

from wireseg.common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    dataset_identity,
    inference_config,
    load_checkpoint,
    read_config,
    resolve_device,
    seed_everything,
    sha256,
    split_paths,
    validate_config,
)
from wireseg.dataset import WireDataset
from wireseg.evaluation import evaluate_model, reuse_predictions
from wireseg.modeling import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--trained-model",
        "--trained_model",
        dest="trained_model",
        default=str(PROJECT_ROOT / "runs/wire700/checkpoint_best_mask.pth"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", "--eval_output_dir", dest="output_dir")
    parser.add_argument(
        "--score-threshold", "--score_threshold", dest="score_threshold", type=float
    )
    parser.add_argument("--max-images", "--max_images", dest="max_images", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--reuse-predictions",
        "--reuse_predictions",
        dest="reuse_predictions",
        action="store_true",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    checkpoint_path = Path(args.trained_model).resolve()
    if not checkpoint_path.is_file():
        parser.error(
            f"Checkpoint does not exist: {checkpoint_path}. Run train.py first."
        )
    checkpoint = load_checkpoint(checkpoint_path)
    identity = dataset_identity(config)
    if identity != checkpoint["dataset_identity"]:
        raise ValueError(
            "Current dataset files/paths differ from the training checkpoint."
        )
    config = inference_config(checkpoint, config)
    if args.score_threshold is not None:
        config["evaluation"]["score_threshold"] = args.score_threshold
    validate_config(config)
    device = resolve_device(args.device)
    seed_everything(config["training"]["seed"])
    annotations, images = split_paths(config, args.split)
    dataset = WireDataset(
        annotations, images, include_targets=False, max_images=args.max_images
    )
    output = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "results" / args.split
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "split": args.split,
        "training_epoch": checkpoint["epoch"],
        "smoke_test": checkpoint["smoke_test"],
        "limited_evaluation": args.max_images is not None,
    }
    if checkpoint["smoke_test"] or args.max_images is not None:
        print(
            "NOTE: smoke/limited evaluation is a code check, not a paper baseline result.",
            flush=True,
        )
    if args.reuse_predictions:
        reuse_predictions(dataset, config, output, metadata)
    else:
        model = build_model(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        evaluate_model(
            model,
            dataset,
            config,
            device,
            output,
            metadata=metadata,
            workers=args.workers,
        )
    print("Saved metrics:", output / "maskrcnn_metrics.json")


if __name__ == "__main__":
    main()
