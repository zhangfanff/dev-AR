"""End-to-end software test on temporary synthetic data, never on invented author labels."""

from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from wirelab.io import ROOT, write, read, sha
from check_kotlin_parity import line_mask


def main():
    parent = ROOT / "validation"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(parent)) as temporary:
        root = Path(temporary)
        assert root.resolve().parent == parent.resolve()
        items = []
        records = {}
        for split, role in [("valid", "dev"), ("test", "test")]:
            data = {
                "info": {},
                "images": [],
                "annotations": [],
                "categories": [{"id": 0, "name": "cable", "supercategory": "cable"}],
            }
            for i, angle in enumerate((-30, 0, 30)):
                key = f"{split}_{i:06d}"
                mask = line_mask(angle)
                image = np.full((128, 128, 3), 235, np.uint8)
                image[mask] = 40
                path = root / "dataset/images" / (key + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(path)
                item = {
                    "key": key,
                    "role": role,
                    "source": "synthetic",
                    "image_id": i,
                    "original_split": split,
                    "width": 128,
                    "height": 128,
                    "image_path": str(path.relative_to(root)).replace("\\", "/"),
                    "image_sha256": sha(path),
                }
                if split == "test" and i == 2:
                    item.update(original_split="external", has_gt_mask=False)
                items.append(item)
                data["images"].append(
                    {"id": i, "width": 128, "height": 128, "file_name": path.name}
                )
                rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                annotation = {
                    "id": i,
                    "image_id": i,
                    "category_id": 0,
                    "segmentation": rle,
                    "iscrowd": 0,
                    "area": float(mask.sum()),
                    "bbox": [0, 0, 128, 128],
                }
                data["annotations"].append(annotation)
                records[key] = [dict(annotation, score=0.9)]
                vector = np.array(
                    [-np.sin(np.deg2rad(angle)), -np.cos(np.deg2rad(angle))]
                )
                center = np.array([64.0, 64.0])
                lines = []
                for region, sign in [("upper", 1), ("lower", -1)]:
                    lines.append(
                        {
                            "region": region,
                            "group": 1,
                            "points": [
                                (center + sign * vector * 5).tolist(),
                                (center + sign * vector * 50).tolist(),
                            ],
                            "directed": True,
                        }
                    )
                write(
                    root / "annotations" / (key + ".json"),
                    {
                        "key": key,
                        "image_sha256": sha(path),
                        "reviewed": True,
                        "regions": {"upper": "labeled", "lower": "labeled"},
                        "lines": lines,
                        "tags": ["synthetic_fixture_only"],
                    },
                )
            write(root / f"dataset/coco_{split}.json", data)
        write(
            root / "dataset/manifest.json",
            {"images": items, "prediction_checkpoint_sha256": "synthetic_only"},
        )
        write(root / "dataset/predictions/synthetic.json", {"records": records})
        commands = [["--role", "dev"], ["--freeze"], ["--role", "test"]]
        for extra in commands:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_experiments.py"),
                    "--root",
                    str(root),
                    "--bootstrap",
                    "40",
                    "--mask-source",
                    "both",
                ]
                + extra,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            print(result.stdout.strip())
        results = sorted((root / "results").iterdir())
        assert len(results) == 2
        for folder in results:
            summary = read(folder / "summary.json")
            assert {r["method"] for r in summary} == {"original", "probabilistic"}
            assert (
                (folder / "summary.csv").is_file()
                and (folder / "sensitivity.png").is_file()
                and (folder / "uncertainty.png").is_file()
            )
            assert len({r["variant"] for r in summary}) >= 10
            if folder.name.startswith("test_"):
                primary = [
                    r
                    for r in summary
                    if r["method"] == "original"
                    and r["variant"] == "default"
                    and r["group"] == "all"
                ]
                assert {r["mask_source"]: r["images"] for r in primary} == {
                    "pred": 3,
                    "gt": 2,
                }
        write(
            ROOT / "validation/pipeline_check.json",
            {
                "passed": True,
                "fixture": "Temporary synthetic mathematical lines only; removed after software checks.",
                "checked": "Complete development, freeze, final-test flow; both algorithms, masks, sensitivity sweeps, CSV, JSON and PNG outputs.",
                "author_annotations_created": False,
                "paper_performance_claim": False,
            },
        )
    print("Direction pipeline tests passed.")


if __name__ == "__main__":
    main()
