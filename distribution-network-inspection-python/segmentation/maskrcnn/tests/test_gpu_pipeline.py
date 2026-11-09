"""Targeted GPU forward/backward checks on negative, zero-width and subpixel annotations."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wireseg.common import (
    PROJECT_ROOT,
    load_checkpoint,
    read_config,
    resolve_device,
    seed_everything,
    split_paths,
    write_json,
)
from wireseg.dataset import WireDataset
from wireseg.modeling import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "runs/smoke_check/checkpoint_best_mask.pth"),
    )
    args = parser.parse_args()
    config = read_config()
    device = resolve_device("cuda")
    seed_everything(42)
    dataset = WireDataset(*split_paths(config, "train"))
    model = build_model(config)
    model.load_state_dict(load_checkpoint(args.checkpoint)["model"])
    model.to(device).train()
    negative_id = next(
        i for i in dataset.image_ids if not dataset.coco.getAnnIds(imgIds=[i])
    )
    image_ids = [
        negative_id,
        dataset.coco.anns[7149]["image_id"],
        dataset.coco.anns[6178]["image_id"],
    ]
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for image_id in image_ids:
        model.zero_grad(set_to_none=True)
        image, target = dataset[dataset.image_ids.index(image_id)]
        with torch.autocast("cuda"):
            losses = model(
                [image.to(device)], [{k: v.to(device) for k, v in target.items()}]
            )
            loss = sum(losses.values())
        assert torch.isfinite(loss), losses
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        row = {
            "image_id": image_id,
            "instances": len(target["labels"]),
            "repaired_annotation_ids": target["repaired_annotation_ids"].tolist(),
            "empty_mask_annotation_ids": target["empty_mask_annotation_ids"].tolist(),
            "losses": {key: value.detach().item() for key, value in losses.items()},
        }
        rows.append(row)
        print("PASS GPU forward/backward:", row, flush=True)
    write_json(
        PROJECT_ROOT / "results/gpu_pipeline_check.json",
        {
            "checks": rows,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "note": "Forward/backward checks only; no optimizer updates or paper results.",
        },
    )


if __name__ == "__main__":
    main()
