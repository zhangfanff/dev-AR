"""Run direction experiments only against explicitly reviewed human annotations."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
import csv
import hashlib
from pathlib import Path
import numpy as np
from wirelab.io import ROOT, manifest, read, write, sha, MaskStore, references
from wirelab import legacy, probabilistic
from wirelab.evaluation import compare, summarize


def code_hashes(root):
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
        for p in sorted((ROOT / "wirelab").glob("*.py"))
    }


def load_labels(root, role, allow_partial=False):
    items = [x for x in manifest(root)["images"] if x["role"] == role]
    labeled = []
    missing = []
    excluded = []
    for item in items:
        path = root / "annotations" / (item["key"] + ".json")
        if not path.exists() or not read(path).get("reviewed", False):
            missing.append(item["key"])
            continue
        label = read(path)
        if label["image_sha256"] != item["image_sha256"]:
            raise ValueError("Annotation image hash mismatch: " + item["key"])
        if label.get("viewpoint", "nadir") in ("oblique", "uncertain"):
            excluded.append(item["key"])
            continue
        if any(s == "unreviewed" for s in label["regions"].values()):
            raise ValueError(
                "Unreviewed region in confirmed annotation: " + item["key"]
            )
        for region, state in label["regions"].items():
            truth = references(label, region)
            if (state == "labeled") != bool(truth):
                raise ValueError(
                    "Region state/reference mismatch: " + item["key"] + " " + region
                )
        labeled.append((item, label))
    if missing and not allow_partial:
        raise RuntimeError(
            f"{role} 已确认有效 {len(labeled)} 张，视角剔除 {len(excluded)} 张，待确认 {len(missing)} 张。请先完成标注；尚未生成实验结果。\n缺少："
            + ", ".join(missing[:12])
        )
    if not labeled:
        raise RuntimeError("没有已确认的人工标注。不会创建空结果或虚构指标。")
    return labeled, missing


def sensitivity_configs(base):
    result = {"default": base}
    for n in (10.0, 20.0, 30.0):
        result[f"bin_{n:g}"] = replace(base, bin_size=n)
    result["bin_offset_7.5"] = replace(base, bin_offset=7.5)
    for n in (1.0, 1.1, 1.4):
        result[f"extend_{n:g}"] = replace(base, extend=n)
    result["no_side_fallback"] = replace(base, side_fallback=False)
    result["no_median_correction"] = replace(base, median_correction=False)
    result["no_horizontal_merge"] = replace(base, merge_horizontal=False)
    for n in (65.0, 85.0):
        result[f"horizontal_{n:g}"] = replace(
            base, horizontal_wrap=n, horizontal_correction=n
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--role", choices=["dev", "test"], default="dev")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument(
        "--freeze",
        action="store_true",
        help="After completing dev annotations, freeze protocol before final test evaluation.",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exploratory runs only; prominently marked incomplete.",
    )
    p.add_argument(
        "--mask-source",
        choices=["both", "gt", "pred"],
        default="pred",
        help="Primary comparison uses frozen predictions; GT diagnostics use only images with real segmentation GT.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Point-estimate debugging only; no uncertainty or sensitivity claims.",
    )
    args = p.parse_args()
    root = args.root.resolve()
    if args.bootstrap < 40:
        raise ValueError("Use at least 40 bootstrap replications (200 by default).")
    if args.freeze:
        dev, _ = load_labels(root, "dev")
        protocol = {
            "schema": "wire_direction_frozen_protocol_v1",
            "baseline": asdict(legacy.Config()),
            "bootstrap": args.bootstrap,
            "seed": 42,
            "max_components": 2,
            "mixture_kappa_cap": 500,
            "selection_manifest_sha256": sha(root / "dataset/manifest.json"),
            "code_sha256": code_hashes(root),
            "dev_annotation_sha256": {
                i["key"]: sha(root / "annotations" / (i["key"] + ".json"))
                for i, _ in dev
            },
            "note": "Parameters fixed using dev images only. Existing splits do not establish route independence.",
        }
        target = root / "frozen_protocol.json"
        if target.exists():
            raise RuntimeError(
                "Protocol already frozen; archive it explicitly before a different development cycle."
            )
        write(target, protocol)
        print("Protocol frozen. Final test labels will not be used for tuning.")
        return
    labeled, missing = load_labels(root, args.role, args.allow_partial)
    base = legacy.Config()
    if args.role == "test":
        frozen = root / "frozen_protocol.json"
        if not frozen.exists():
            raise RuntimeError("先完成开发集并用 --freeze 固定方案，再评测测试集。")
        protocol = read(frozen)
        if protocol["code_sha256"] != code_hashes(root) or protocol[
            "selection_manifest_sha256"
        ] != sha(root / "dataset/manifest.json"):
            raise RuntimeError(
                "Code or selected dataset changed after protocol freeze. Do not tune silently on test results."
            )
        if not args.quick and args.bootstrap != protocol["bootstrap"]:
            raise RuntimeError("Bootstrap count differs from frozen protocol.")
        base = legacy.Config(**protocol["baseline"])
    store = MaskStore(root)
    kinds = ["gt", "pred"] if args.mask_source == "both" else [args.mask_source]
    for item, _ in labeled:
        if "pred" in kinds and item["key"] not in store.pred:
            raise RuntimeError(
                "缺少预测缓存，请运行 prepare_predictions.py（默认CPU）。"
            )
    rows = []
    predictions = {}
    configs = sensitivity_configs(base) if not args.quick else {"default": base}
    for it, (item, label) in enumerate(labeled):
        seed = int(hashlib.sha256(item["key"].encode()).hexdigest()[:8], 16) + 42
        for kind in kinds:
            if kind == "gt" and not item.get("has_gt_mask", True):
                continue
            masks = store.load(item, kind)
            w, h = item["width"], item["height"]
            results = legacy.analyze(masks, w, h, base)
            predictions[f"{item['key']}/{kind}/original"] = results
            rows.extend(compare(item, label, results, "original", kind))
            results = probabilistic.analyze(
                masks,
                w,
                h,
                base,
                bootstrap=0 if args.quick else args.bootstrap,
                seed=seed,
            )
            predictions[f"{item['key']}/{kind}/probabilistic"] = results
            rows.extend(compare(item, label, results, "probabilistic", kind))
            for name, cfg in configs.items():
                if name == "default":
                    continue
                result = legacy.analyze(masks, w, h, cfg)
                rows.extend(
                    compare(item, label, result, "original", kind, variant=name)
                )
        print(it + 1, len(labeled), item["key"], flush=True)
    if not rows:
        raise RuntimeError("所有已标区域均无法判定；没有可评测数据，不生成性能结果。")
    summary = []
    keys = sorted({(r["method"], r["mask_source"], r["variant"]) for r in rows})
    for method, kind, variant in keys:
        selected = [
            r
            for r in rows
            if (r["method"], r["mask_source"], r["variant"]) == (method, kind, variant)
        ]
        for group in ["all"] + sorted({r["source"] for r in selected}):
            subset = (
                selected
                if group == "all"
                else [r for r in selected if r["source"] == group]
            )
            if subset:
                summary.append(
                    dict(
                        method=method,
                        mask_source=kind,
                        variant=variant,
                        group=group,
                        **summarize(subset),
                    )
                )
        tags = sorted({t for r in selected for t in r["tags"]})
        for tag in tags:
            subset = [r for r in selected if tag in r["tags"]]
            summary.append(
                dict(
                    method=method,
                    mask_source=kind,
                    variant=variant,
                    group="tag:" + tag,
                    **summarize(subset),
                )
            )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = (
        root
        / "results"
        / (args.role + "_" + stamp + ("_EXPLORATORY" if missing or args.quick else ""))
    )
    out.mkdir(parents=True)
    meta = {
        "role": args.role,
        "selected_images": len(labeled),
        "missing_annotations": missing,
        "viewpoint_excluded": {
            i["key"]: read(root / "annotations" / (i["key"] + ".json")).get("viewpoint")
            for i in manifest(root)["images"]
            if i["role"] == args.role
            and (root / "annotations" / (i["key"] + ".json")).exists()
            and read(root / "annotations" / (i["key"] + ".json")).get("viewpoint")
            in ("oblique", "uncertain")
        },
        "gt_mask_available_images": sum(i.get("has_gt_mask", True) for i, _ in labeled),
        "gt_mask_unavailable_images": [
            i["key"] for i, _ in labeled if not i.get("has_gt_mask", True)
        ],
        "segmentation_training_seen_images": [
            i["key"] for i, _ in labeled if i.get("segmentation_training_seen", False)
        ],
        "exploratory": bool(missing or args.quick),
        "quick_mode": args.quick,
        "bootstrap": args.bootstrap,
        "baseline_config": asdict(base),
        "code_sha256": code_hashes(root),
        "manifest_sha256": sha(root / "dataset/manifest.json"),
        "annotation_sha256": {
            i["key"]: sha(root / "annotations" / (i["key"] + ".json"))
            for i, _ in labeled
        },
        "prediction_cache_sha256": {
            p.name: sha(p)
            for p in sorted((root / "dataset/predictions").glob("*.json"))
        },
        "region_status_counts": dict(
            Counter(s for _, label in labeled for s in label["regions"].values())
        ),
        "interval_scope": "Approximate 95% conditional spatial-block/instance bootstrap intervals. Membership and segmentation model are fixed; no nominal coverage guarantee.",
        "metric_scope": "Image-plane axes and separately annotated outgoing arrows; not aircraft yaw or flight-control error.",
    }
    write(out / "run_metadata.json", meta)
    write(out / "per_region.json", rows)
    write(out / "predictions.json", predictions)
    write(out / "summary.json", summary)
    fields = list(summary[0])
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    from wirelab.plots import plot_results

    plot_results(summary, out)
    print("Saved actual annotation-based results:", out)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print("实验尚未运行：" + str(error))
        raise SystemExit(2)
