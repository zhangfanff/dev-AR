from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from pycocotools import mask as mask_api

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def manifest(root=ROOT):
    return read(Path(root) / "dataset/manifest.json")


def decode(seg, h, w):
    if isinstance(seg, list):
        seg = mask_api.merge(mask_api.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        seg = mask_api.frPyObjects(seg, h, w)
    else:
        seg = copy.deepcopy(seg)
        if isinstance(seg["counts"], str):
            seg["counts"] = seg["counts"].encode("ascii")
    result = mask_api.decode(seg)
    if result.ndim == 3:
        result = result.any(axis=2)
    return result.astype(bool)


class MaskStore:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.m = manifest(root)
        self.gt = {}
        for split in sorted(
            {
                x["original_split"]
                for x in self.m["images"]
                if x.get("has_gt_mask", True)
            }
        ):
            d = read(self.root / f"dataset/coco_{split}.json")
            by = {}
            for a in d["annotations"]:
                by.setdefault(a["image_id"], []).append(a)
            self.gt[split] = by
        self.pred = {}
        for p in (self.root / "dataset/predictions").glob("*.json"):
            d = read(p)
            saved = d.get("metadata", {}).get("checkpoint_sha256")
            if saved and saved != self.m["prediction_checkpoint_sha256"]:
                raise ValueError("Wrong prediction checkpoint: " + str(p))
            if "records" in d:
                self.pred.update(d["records"])

    def load(self, item, kind):
        if kind == "gt":
            if not item.get("has_gt_mask", True):
                raise ValueError(
                    "该现场照片没有分割真值掩码；不能把缺失标注当作空图：" + item["key"]
                )
            rows = self.gt[item["original_split"]].get(item["image_id"], [])
        else:
            if item["key"] not in self.pred:
                raise FileNotFoundError(
                    "缺少预测缓存：" + item["key"] + "。先运行 prepare_predictions.py。"
                )
            rows = self.pred[item["key"]]
        rows = [a for a in rows if a["category_id"] == 0]
        return [
            (
                decode(a["segmentation"], item["height"], item["width"]),
                float(a.get("score", 1.0)),
            )
            for a in rows
        ]


def axial_delta(a, b):
    return (np.asarray(a) - b + 90.0) % 180.0 - 90.0


def directed_delta(a, b):
    return (np.asarray(a) - b + 180.0) % 360.0 - 180.0


def axial_mean(angles, weights=None):
    a = np.deg2rad(np.asarray(angles) * 2)
    w = np.ones(len(a)) if weights is None else np.asarray(weights)
    return float(
        np.rad2deg(np.arctan2(np.sum(w * np.sin(a)), np.sum(w * np.cos(a)))) / 2
    )


def references(annotation, region):
    grouped = {}
    for line in annotation.get("lines", []):
        if line["region"] != region:
            continue
        p = np.asarray(line["points"], dtype=float)
        d = np.diff(p, axis=0)
        length = np.linalg.norm(d, axis=1)
        keep = length > 1e-6
        if not keep.any():
            continue
        angles = np.rad2deg(np.arctan2(-d[keep, 0], -d[keep, 1]))
        group = grouped.setdefault(
            str(line["group"]), {"angles": [], "weights": [], "directed": True}
        )
        group["angles"].extend(angles.tolist())
        group["weights"].extend(length[keep].tolist())
        group["directed"] &= line.get("directed", False)
    result = []
    for group, g in grouped.items():
        theta = axial_mean(g["angles"], g["weights"])
        directed = None
        if g["directed"]:
            a = np.deg2rad(g["angles"])
            w = np.asarray(g["weights"])
            directed = float(
                np.rad2deg(np.arctan2(np.sum(w * np.sin(a)), np.sum(w * np.cos(a))))
            )
        result.append({"group": group, "angle": theta, "direction": directed})
    return result
