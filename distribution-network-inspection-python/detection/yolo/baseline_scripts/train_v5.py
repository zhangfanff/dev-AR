import yaml
from common import PROJECT, SCRIPTS, parser, preparation
from v5_runtime import activate


def main():
    args = parser("yolov5n", "train").parse_args()
    report = preparation(args, "yolov5n", "train")
    if args.dry_run:
        return
    vendor = activate()
    from pathlib import Path

    weights = Path(report["weights"])
    if not weights.is_file():
        if args.weights:
            raise FileNotFoundError(weights)
        raise FileNotFoundError(
            "Pretrained yolov5n.pt missing; run prepare_detector_baselines.py first."
        )
    settings = report["settings"]
    hyp = yaml.safe_load((vendor / "data/hyps/hyp.scratch-low.yaml").read_text())
    for key in [
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "copy_paste",
        "bgr",
        "close_mosaic",
    ]:
        hyp[key] = settings[key]
    hyp_file = SCRIPTS / "configs" / f"{args.task}_yolov5_hyp.yaml"
    hyp_file.write_text(yaml.safe_dump(hyp, sort_keys=False), encoding="utf-8")
    import train

    # Keep logging local; no external experiment tracking.
    init = train.Loggers.__init__

    def local_logging(self, *a, **kw):
        kw["include"] = ("csv", "tb")
        return init(self, *a, **kw)

    train.Loggers.__init__ = local_logging
    train.run(
        data=report["data"],
        weights=str(weights),
        hyp=str(hyp_file),
        epochs=settings["epochs"],
        batch_size=settings["batch"],
        imgsz=640,
        optimizer="SGD",
        device=args.device,
        workers=args.workers,
        seed=0,
        cos_lr=False,
        patience=100,
        project=str(PROJECT / "runs/detect"),
        name=report["name"],
        exist_ok=False,
        resume=False,
        cache=None,
        noplots=False,
    )


if __name__ == "__main__":
    main()
