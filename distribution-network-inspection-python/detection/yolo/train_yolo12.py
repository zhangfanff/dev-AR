"""Train YOLOv12n for either detection task using shared task settings."""

from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent / "baseline_scripts"
sys.path.insert(0, str(SCRIPTS))
from common import PROJECT, parser, preparation


def main():
    args = parser("yolov12n", "train").parse_args()
    report = preparation(args, "yolov12n", "train")
    if args.dry_run:
        return
    from dependencies import activate

    activate("yolov12")
    from ultralytics import YOLO

    weights = Path(report["weights"])
    if args.weights and not weights.is_file():
        raise FileNotFoundError(weights)
    model = YOLO(str(weights) if weights.is_file() else "yolov12n.pt")
    model.train(
        data=report["data"],
        device=args.device,
        workers=args.workers,
        project=str(PROJECT / "runs/detect"),
        name=report["name"],
        exist_ok=False,
        resume=False,
        pretrained=True,
        **report["settings"],
    )


if __name__ == "__main__":
    main()
