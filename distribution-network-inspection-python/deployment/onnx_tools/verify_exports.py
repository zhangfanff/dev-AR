"""Compare exported ONNX tensors with the selected PyTorch checkpoint on a real test image."""

from pathlib import Path
import argparse, json, sys
import numpy as np
from PIL import Image
from prepare_models import ROOT, YOLO, WEIGHTS, sha
from evaluate_resolutions import preprocess, dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=ROOT / "models")
    a = p.parse_args()
    import torch, onnxruntime as ort

    torch.set_num_threads(4)
    reports = []
    # One default-size export per task; other sizes additionally covered by smoke tests.
    for task, size in [("tower", 640), ("tower_top", 640), ("wire", 648)]:
        folder = a.models / f"{task}_{size}"
        config = json.loads((folder / "model.json").read_text())
        assert sha(folder / "model.onnx") == config["onnx_sha256"]
        gt, paths = dataset(task)
        with Image.open(paths[gt["images"][0]["id"]]) as image:
            x, _ = preprocess(image.convert("RGB"), config)
        if task == "wire":
            from rfdetr import RFDETRSegPreview

            detector = RFDETRSegPreview(
                pretrain_weights=str(WEIGHTS[task]),
                resolution=size,
                num_select=200,
                device="cpu",
            )
            model = detector.model.model.eval()
            model.export()
        else:
            sys.path.insert(0, str(YOLO))
            from ultralytics import YOLO as Detector

            model = Detector(str(WEIGHTS[task])).model.eval()
            model.fuse(verbose=False)
        with torch.no_grad():
            raw = model(torch.from_numpy(x))
        expected = (
            list(raw) if task == "wire" else [raw[0] if isinstance(raw, tuple) else raw]
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(folder / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        actual = session.run(config["outputs"], {config["input"]: x})
        row = dict(
            task=task,
            resolution=size,
            checkpoint_sha256=config["checkpoint_sha256"],
            onnx_sha256=config["onnx_sha256"],
            outputs=[],
        )
        for name, e, v in zip(config["outputs"], expected, actual):
            e = e.detach().cpu().numpy()
            assert v.shape == e.shape
            item = dict(
                name=name,
                shape=list(v.shape),
                max_abs_error=float(np.max(np.abs(v - e))),
                mean_abs_error=float(np.mean(np.abs(v - e))),
                strict_allclose_1e3=bool(np.allclose(v, e, rtol=1e-3, atol=1e-3)),
            )
            if v.ndim == 4:
                # Large mask logits can differ locally between attention kernels.
                # Keep the strict check result, and additionally check actual binary-mask stability.
                item["binary_mask_agreement"] = float(np.mean((v > 0) == (e > 0)))
                item["criterion"] = (
                    "mean absolute logit error < 0.001 and thresholded-mask agreement > 99.9%"
                )
                assert (
                    item["mean_abs_error"] < 0.001
                    and item["binary_mask_agreement"] > 0.999
                ), item
            else:
                np.testing.assert_allclose(v, e, rtol=1e-3, atol=1e-3)
            row["outputs"].append(item)
        reports.append(row)
        print(task, "ONNX/PyTorch parity passed", flush=True)
        del model, session
    (ROOT / "export_verification.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
