"""Generate segmentation caches for the configured direction dataset."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from wirelab.io import ROOT, manifest, read, write, sha, MaskStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    store = MaskStore(args.root)
    items = [x for x in store.m["images"] if x["key"] not in store.pred]
    if not items:
        print("All selected images already have prediction caches.")
        return
    checkpoint = (
        ROOT.parent / "segmentation/rfdetr/model_new648/checkpoint_best_total.pth"
    )
    if sha(checkpoint) != store.m["prediction_checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint changed.")
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(42)
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview(
        pretrain_weights=str(checkpoint),
        resolution=648,
        num_select=200,
        device=args.device,
    )
    target = args.root / "dataset/predictions/additional.json"
    records = read(target)["records"] if target.exists() else {}
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "device": args.device,
        "resolution": 648,
        "score": 0.3,
        "num_select": 200,
        "max_cable_predictions": 100,
    }
    for i, item in enumerate(items):
        image = np.asarray(Image.open(args.root / item["image_path"]).convert("RGB"))
        with torch.inference_mode():
            pred = model.predict(image, threshold=0.3)
        rows = []
        keep = np.flatnonzero(pred.class_id == 0)
        keep = keep[np.argsort(-pred.confidence[keep], kind="stable")[:100]]
        for j in keep:
            rle = mask_api.encode(np.asfortranarray(pred.mask[j].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            rows.append(
                {
                    "category_id": 0,
                    "segmentation": rle,
                    "score": float(pred.confidence[j]),
                }
            )
        records[item["key"]] = rows
        write(target, {"metadata": metadata, "records": records})
        print(i + 1, len(items), item["key"], len(rows), flush=True)
    write(target, {"metadata": metadata, "records": records})


if __name__ == "__main__":
    main()
