"""Explicit RF-DETR SEG training entry; datasets and initialization are external."""

import argparse
import multiprocessing
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=648)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(args.resolution, args.epochs, args.batch, args.accumulation) <= 0
        or args.lr <= 0
        or args.workers < 0
    ):
        parser.error(
            "Training sizes and learning rate must be positive; workers must be nonnegative"
        )
    if not args.dataset.is_dir() or not args.pretrained.is_file():
        parser.error("Provide an existing dataset directory and pretrained checkpoint")
    if args.output.exists():
        parser.error("Use a new output directory to avoid overwriting runs")
    if args.dry_run:
        print(vars(args))
        return
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview(
        pretrain_weights=str(args.pretrained), resolution=args.resolution
    )
    model.train(
        dataset_dir=str(args.dataset.resolve()),
        output_dir=str(args.output.resolve()),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accumulation,
        lr=args.lr,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
