"""Inspect human annotation progress and geometry without estimating performance."""

from pathlib import Path
from collections import Counter
import numpy as np
from wirelab.io import ROOT, manifest, read, references


def main():
    totals = Counter()
    done = Counter()
    errors = []
    states = Counter()
    excluded = Counter()
    for item in manifest()["images"]:
        category = item["role"] + "/" + item["source"]
        totals[category] += 1
        path = ROOT / "annotations" / (item["key"] + ".json")
        if not path.exists():
            continue
        label = read(path)
        if not label.get("reviewed"):
            continue
        done[category] += 1
        if label.get("image_sha256") != item["image_sha256"]:
            errors.append(item["key"] + ": image hash")
        if label.get("viewpoint", "nadir") in ("oblique", "uncertain"):
            excluded[label["viewpoint"]] += 1
            continue
        for region, state in label["regions"].items():
            states[state] += 1
            refs = references(label, region)
            if state == "unreviewed" or ((state == "labeled") != bool(refs)):
                errors.append(item["key"] + ": inconsistent " + region)
        for line in label["lines"]:
            points = np.asarray(line["points"], dtype=float)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or len(points) < 2
                or not np.isfinite(points).all()
            ):
                errors.append(item["key"] + ": invalid points")
                continue
            if (
                (points < 0).any()
                or (points[:, 0] >= item["width"]).any()
                or (points[:, 1] >= item["height"]).any()
            ):
                errors.append(item["key"] + ": outside image")
            if (
                line["region"] == "upper"
                and (points[:, 1] >= 0.6 * item["height"]).any()
            ) or (
                line["region"] == "lower"
                and (points[:, 1] < 0.4 * item["height"]).any()
            ):
                errors.append(item["key"] + ": outside selected region")
    for category, n in sorted(totals.items()):
        print(category, done[category], "/", n, "confirmed")
    print("Region status:", dict(states))
    print("Excluded viewpoints (not negative examples):", dict(excluded))
    if errors:
        raise RuntimeError("\n".join(errors))
    print(
        "No consistency errors in confirmed annotations. No performance metrics generated."
    )


if __name__ == "__main__":
    main()
