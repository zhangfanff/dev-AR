"""Export an external RF-DETR SEG checkpoint without top-level execution."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    if args.output.exists():
        parser.error("Use a new output directory")
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview(pretrain_weights=str(args.checkpoint.resolve()))
    model.export(output_dir=str(args.output.resolve()))


if __name__ == "__main__":
    main()
