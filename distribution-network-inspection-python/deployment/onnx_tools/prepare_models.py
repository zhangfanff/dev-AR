from pathlib import Path as _DependencyPath
import sys as _dependency_sys

_dependency_sys.path.insert(0, str(_DependencyPath(__file__).resolve().parents[2]))
from dependencies import activate as _activate_dependency

_activate_dependency("yolov12")

"""Export static FP32 ONNX models and preprocessing metadata."""

from pathlib import Path
import argparse, hashlib, json, shutil, sys

ROOT = Path(__file__).resolve().parents[1]
YOLO = Path(__file__).resolve().parents[2] / "detection/yolo"
RF = Path(__file__).resolve().parents[2] / "segmentation/rfdetr"
WEIGHTS = {
    "tower": YOLO / "runs/detect/tower/weights/best.pt",
    "tower_top": YOLO / "runs/detect/train/weights/best.pt",
    "wire": RF / "model_new648/checkpoint_best_total.pth",
}
NAMES = {
    "tower": ["tower"],
    "tower_top": ["tower-top", "hengdan", "jinggai", "car", "byq"],
    "wire": ["cable"],
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sidecar(folder, task, weight, size):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(folder / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    inp = session.get_inputs()[0]
    outputs = session.get_outputs()
    assert inp.shape == [1, 3, size, size], inp.shape
    assert len(outputs) == (3 if task == "wire" else 1), [
        (o.name, o.shape) for o in outputs
    ]
    if task == "wire":
        assert outputs[1].shape[-1] == 1, outputs[1].shape
    else:
        assert outputs[0].shape[1] == 4 + len(NAMES[task]), outputs[0].shape
    config = dict(
        task=task,
        file="model.onnx",
        width=size,
        height=size,
        classes=len(NAMES[task]),
        class_names=NAMES[task],
        input=inp.name,
        outputs=[o.name for o in outputs],
        output_shapes=[o.shape for o in outputs],
        onnx_sha256=sha(folder / "model.onnx"),
        checkpoint=str(weight),
        checkpoint_sha256=sha(weight),
        preprocessing="stretch_rgb_imagenet"
        if task == "wire"
        else "letterbox_rgb_114_div255",
        confidence=0.3 if task == "wire" else 0.25,
        iou=0.7,
        max_det=100,
        mask_postprocess="bilinear logits to original size, align_corners=false, threshold zero",
    )
    (folder / "model.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(folder, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", choices=list(WEIGHTS), default=list(WEIGHTS))
    p.add_argument("--yolo-sizes", nargs="+", type=int, default=[320, 480, 640])
    p.add_argument("--wire-sizes", nargs="+", type=int, default=[432, 576, 648])
    p.add_argument("--output", type=Path, default=ROOT / "models")
    args = p.parse_args()
    import torch

    torch.set_num_threads(4)
    for task in args.tasks:
        weight = WEIGHTS[task]
        if not weight.is_file():
            raise FileNotFoundError(weight)
        for size in args.wire_sizes if task == "wire" else args.yolo_sizes:
            if size % (24 if task == "wire" else 32):
                raise ValueError(
                    "Size must be divisible by patch/window factor 24 (wire) or stride 32 (YOLO)"
                )
            folder = args.output / f"{task}_{size}"
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "model.onnx").exists():
                raise FileExistsError(f"Use a fresh output folder: {folder}")
            if task == "wire":
                from rfdetr import RFDETRSegPreview

                model = RFDETRSegPreview(
                    pretrain_weights=str(weight),
                    resolution=size,
                    num_select=200,
                    device="cpu",
                )
                assert model.model.class_names == ["cable"]
                model.export(output_dir=str(folder), simplify=False, verbose=False)
                (folder / "inference_model.onnx").rename(folder / "model.onnx")
                del model
            else:
                sys.path.insert(0, str(YOLO))
                from ultralytics import YOLO as Detector

                model = Detector(str(weight))
                assert list(model.names.values()) == NAMES[task], model.names
                # Export a private checkpoint copy: never alter users' runs/ directories.
                local = folder / "source_checkpoint.pt"
                shutil.copy2(weight, local)
                exported = Detector(str(local)).export(
                    format="onnx",
                    imgsz=size,
                    batch=1,
                    dynamic=False,
                    simplify=False,
                    opset=17,
                    half=False,
                    nms=False,
                    device="cpu",
                )
                Path(exported).rename(folder / "model.onnx")
                local.unlink()
            sidecar(folder, task, weight, size)
    print(
        "Push the generated folders to /sdcard/Android/data/com.inspection.onnxbenchmark/files/models/"
    )


if __name__ == "__main__":
    main()
