from pathlib import Path
import sys
from common import PROJECT, parser, preparation


def main():
    args = parser("yolov8n", "train").parse_args()
    report = preparation(args, "yolov8n", "train")
    if args.dry_run:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dependencies import activate

    activate("yolov12")
    from ultralytics import YOLO

    weights = Path(report["weights"])
    if args.weights and not weights.is_file():
        raise FileNotFoundError(weights)
    model = YOLO(str(weights) if weights.is_file() else "yolov8n.pt")
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
