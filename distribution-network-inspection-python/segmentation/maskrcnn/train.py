"""Train on train.json, select by validation mask AP, never select on test.json."""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wireseg.common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    atomic_checkpoint,
    dataset_identity,
    environment,
    load_checkpoint,
    read_config,
    resolve_device,
    seed_everything,
    seed_worker,
    split_paths,
    validate_config,
    write_json,
)
from wireseg.dataset import WireDataset, collate_fn
from wireseg.evaluation import evaluate_model
from wireseg.modeling import build_model


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def comparable_config(config):
    result = copy.deepcopy(config)
    result["training"].pop(
        "epochs"
    )  # Extending the planned number of epochs is permitted.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "runs/wire700"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume", help="Resume a checkpoint_last.pth from this project."
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initialization", choices=["COCO_V1", "random"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch, two training batches, three val images.",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-images", type=int)
    args = parser.parse_args()
    previous = load_checkpoint(args.resume) if args.resume else None
    config = (
        copy.deepcopy(previous["config"])
        if previous and not args.config
        else read_config(args.config or DEFAULT_CONFIG)
    )
    train = config["training"]
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "workers": args.workers,
        "learning_rate": args.lr,
        "seed": args.seed,
        "initialization": args.initialization,
    }
    for key, value in overrides.items():
        if value is not None:
            train[key] = value
    if args.no_amp:
        train["amp"] = False
    if args.smoke:
        train["epochs"] = args.epochs or 1
        args.max_train_batches = args.max_train_batches or 2
        args.max_val_images = args.max_val_images or 3
    for value in (args.max_train_batches, args.max_val_images):
        if value is not None and value < 1:
            parser.error("Sample/batch limits must be positive.")
    if train["initialization"] not in ("COCO_V1", "random"):
        parser.error("Unknown initialization.")
    validate_config(config)
    smoke = (
        args.smoke
        or args.max_train_batches is not None
        or args.max_val_images is not None
    )
    if previous and (
        smoke != previous["smoke_test"]
        or {
            "max_train_batches": args.max_train_batches,
            "max_val_images": args.max_val_images,
        }
        != previous["limits"]
    ):
        raise ValueError("Resume requires the same full/smoke mode and data limits.")
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if previous and Path(args.resume).resolve().parent != run:
        raise ValueError(
            "Resume in the original --run-dir so its best checkpoint and logs stay together."
        )
    if not previous and (run / "checkpoint_last.pth").exists():
        parser.error("Run already has a checkpoint; use --resume or a new --run-dir.")
    identity = dataset_identity(config)
    if previous:
        if previous["dataset_identity"] != identity:
            raise ValueError("Cannot resume with changed dataset files or paths.")
        if comparable_config(previous["config"]) != comparable_config(config):
            raise ValueError("Resume settings differ; only --epochs may change.")
    seed_everything(train["seed"])
    device = resolve_device(args.device)
    training_set = WireDataset(
        *split_paths(config, "train"),
        training=True,
        horizontal_flip_probability=train["horizontal_flip_probability"],
    )
    validation_set = WireDataset(
        *split_paths(config, "val"),
        include_targets=False,
        max_images=args.max_val_images,
    )
    generator = torch.Generator()
    loader = DataLoader(
        training_set,
        batch_size=train["batch_size"],
        shuffle=True,
        num_workers=train["workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    batches = min(len(loader), args.max_train_batches or len(loader))
    model = build_model(
        config,
        initialize_pretrained=not previous and train["initialization"] == "COCO_V1",
    )
    if previous:
        model.load_state_dict(previous["model"], strict=True)
    model.to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=train["learning_rate"],
        momentum=train["momentum"],
        weight_decay=train["weight_decay"],
    )
    amp_enabled = train["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=train.get("amp_initial_scale", 4096.0), enabled=amp_enabled
    )
    start_epoch, global_step, best_mask_ap = 1, 0, -math.inf
    if previous:
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        start_epoch = previous["epoch"] + 1
        global_step = previous["global_step"]
        best_mask_ap = previous["best_mask_AP"]
        restore_rng(previous["rng_state"])
    if start_epoch > train["epochs"]:
        parser.error("Checkpoint already reached --epochs; increase it to continue.")
    write_json(run / "config.json", config)
    write_json(
        run / "run_metadata.json",
        {
            "config": config,
            "dataset_identity": identity,
            "environment": environment(),
            "smoke_test": smoke,
            "selection_metric": "validation mask AP",
            "test_set_used_for_selection": False,
        },
    )
    print(
        f"Train images={len(training_set)}, val images={len(validation_set)}, "
        f"device={device}, AMP={amp_enabled}, smoke={smoke}",
        flush=True,
    )
    for epoch in range(start_epoch, train["epochs"] + 1):
        generator.manual_seed(train["seed"] + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        skipped_updates = 0
        started = time.perf_counter()
        base_lr = train["learning_rate"] * train["lr_gamma"] ** sum(
            epoch > milestone for milestone in train["lr_milestones"]
        )
        for index, (images, targets) in enumerate(loader):
            if index >= batches:
                break
            group_start = (index // train["accumulation_steps"]) * train[
                "accumulation_steps"
            ]
            group_size = min(train["accumulation_steps"], batches - group_start)
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {
                    key: value.to(device, non_blocking=True)
                    for key, value in target.items()
                }
                for target in targets
            ]
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss: {losses}")
            scaler.scale(loss / group_size).backward()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.detach().item()
            if (index + 1) % train["accumulation_steps"] == 0 or index + 1 == batches:
                warmup = (
                    0.001 + 0.999 * min(global_step / train["warmup_steps"], 1.0)
                    if train["warmup_steps"]
                    else 1.0
                )
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * warmup
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before:
                    global_step += 1
                else:
                    skipped_updates += 1
                    print(
                        "AMP reduced its scale and skipped an overflowing update.",
                        flush=True,
                    )
            if index % 25 == 0 or index + 1 == batches:
                print(
                    f"Epoch {epoch}/{train['epochs']} batch {index + 1}/{batches}, "
                    f"loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.7f}",
                    flush=True,
                )
        # Validation is inference-only: no loss-mode BN updates or training on held-out labels.
        report = evaluate_model(
            model,
            validation_set,
            config,
            device,
            run / f"validation/epoch_{epoch:03d}",
            metadata={
                "split": "val",
                "training_epoch": epoch,
                "smoke_test": smoke,
                "limited_evaluation": args.max_val_images is not None,
                "dataset_identity": identity,
            },
            workers=train["workers"],
        )
        score = report["metrics"]["segm"]["AP"]
        is_best = score > best_mask_ap
        best_mask_ap = max(best_mask_ap, score)
        checkpoint = {
            "format_version": 1,
            "architecture": "maskrcnn_resnet50_fpn",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_mask_AP": best_mask_ap,
            "config": config,
            "dataset_identity": identity,
            "label_to_category": training_set.label_to_category,
            "rng_state": rng_state(),
            "smoke_test": smoke,
            "limits": {
                "max_train_batches": args.max_train_batches,
                "max_val_images": args.max_val_images,
            },
            "validation_metrics": report["metrics"],
            "environment": environment(),
        }
        atomic_checkpoint(run / "checkpoint_last.pth", checkpoint)
        if is_best:
            atomic_checkpoint(run / "checkpoint_best_mask.pth", checkpoint)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": {k: v / batches for k, v in totals.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_box_AP": report["metrics"]["bbox"]["AP"],
            "val_mask_AP": score,
            "best_mask_AP": best_mask_ap,
            "seconds": time.perf_counter() - started,
            "smoke_test": smoke,
            "amp_skipped_updates": skipped_updates,
        }
        with (run / "training_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(
            f"Epoch {epoch} complete: val mask AP={score:.4f}%, "
            f"best={best_mask_ap:.4f}%; checkpoint saved.",
            flush=True,
        )
    print("Training complete. Best checkpoint:", run / "checkpoint_best_mask.pth")
    if smoke:
        print(
            "SMOKE CHECK ONLY: these weights/results are not a completed paper baseline."
        )


if __name__ == "__main__":
    main()
