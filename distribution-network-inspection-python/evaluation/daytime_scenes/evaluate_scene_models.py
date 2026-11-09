# -*- coding: utf-8 -*-
"""Compute detection, segmentation and direction metrics by scene group."""

import contextlib, csv, hashlib, io, json, sys
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
WORK = Path(__file__).resolve().parents[2] / "."
OUT = WORK / "outputs/reviewer2_scene_evaluation_20261103"
LAB = Path(__file__).resolve().parents[2] / "direction"
sys.path.insert(0, str(LAB))
from wirelab.evaluation import summarize


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def mounted(p):
    p = Path(str(p).replace("H:", "I:"))
    assert p.exists(), str(p)
    return p


def evaluate(gt, predictions, kind, threshold, cat=0):
    """COCO AP, fixed-threshold instance P/R, positive-image detection success."""
    image_ids = {im["id"] for im in gt["images"]}
    predictions = [
        deepcopy(p)
        for p in predictions
        if p["image_id"] in image_ids and p["category_id"] == cat
    ]
    gt = deepcopy(gt)
    gt.setdefault("info", {})
    with contextlib.redirect_stdout(io.StringIO()):
        api = COCO()
        api.dataset = gt
        api.createIndex()
        if predictions:
            det = api.loadRes(predictions)
        else:
            det = COCO()
            det.dataset = {**gt, "annotations": []}
            det.createIndex()
        ev = COCOeval(api, det, kind)
        ev.params.imgIds = sorted(image_ids)
        ev.params.catIds = [cat]
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    target_counts = Counter(
        a["image_id"]
        for a in gt["annotations"]
        if a["category_id"] == cat
        and not a.get("iscrowd", 0)
        and not a.get("ignore", 0)
    )
    per_image = []
    for rec in ev.evalImgs:
        if (
            rec is None
            or rec["category_id"] != cat
            or list(rec["aRng"]) != list(ev.params.areaRng[0])
        ):
            continue
        keep = (np.asarray(rec["dtScores"]) >= threshold) & ~rec["dtIgnore"][0]
        tp = int(np.sum(keep & (rec["dtMatches"][0] > 0)))
        fp = int(np.sum(keep & (rec["dtMatches"][0] == 0)))
        n = target_counts[rec["image_id"]]
        assert tp <= n
        per_image.append(
            dict(
                image_id=int(rec["image_id"]),
                targets=int(n),
                TP=tp,
                FP=fp,
                FN=int(n - tp),
                target_present=bool(n),
                success=bool(n and tp),
            )
        )
    covered = {r["image_id"] for r in per_image}
    for iid in sorted(image_ids - covered):
        per_image.append(
            dict(
                image_id=int(iid),
                targets=int(target_counts[iid]),
                TP=0,
                FP=0,
                FN=int(target_counts[iid]),
                target_present=bool(target_counts[iid]),
                success=False,
            )
        )
    assert len(per_image) == len(image_ids)
    tp = sum(r["TP"] for r in per_image)
    fp = sum(r["FP"] for r in per_image)
    targets = sum(target_counts.values())
    positive = sum(bool(target_counts[i]) for i in image_ids)
    successful = sum(r["success"] for r in per_image)
    precision = ev.eval["precision"][:, :, 0, 0, -1]
    ap = [float(p[p >= 0].mean() * 100) if (p >= 0).any() else None for p in precision]
    result = dict(
        images=len(image_ids),
        target_instances=targets,
        positive_images=positive,
        negative_images=len(image_ids) - positive,
        AP_percent=float(ev.stats[0] * 100),
        AP50_percent=float(ev.stats[1] * 100),
        AP75_percent=float(ev.stats[2] * 100),
        P_percent=100 * tp / (tp + fp) if tp + fp else 0.0,
        R_percent=100 * tp / targets if targets else None,
        TP=tp,
        FP=fp,
        FN=targets - tp,
        success_images=successful,
        success_percent=100 * successful / positive if positive else None,
        fixed_confidence=threshold,
        success_iou=0.5,
        eval_kind=kind,
        AP_by_iou_percent=ap,
        success_definition="Positive image with at least one correctly matched target at IoU >=0.50 and fixed confidence; not flight success.",
        background_images_with_false_positive=sum(
            r["FP"] > 0 and not r["target_present"] for r in per_image
        ),
    )
    return result, sorted(per_image, key=lambda r: r["image_id"])


def assert_close(a, b, label):
    assert abs(a - b) < 1e-6, (label, a, b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene_index = load(ROOT / "source_index.json")
    definitions = load(ROOT / "scene_definitions.json")
    evidence = {
        "created_at": datetime.now().isoformat(),
        "scene_root": str(ROOT),
        "scene_index_sha256": sha(ROOT / "source_index.json"),
        "scene_summary_sha256": sha(ROOT / "summary.json"),
        "scope": "Normal-weather daytime distribution-network inspection",
        "checkpoint_selection": "Existing validation-selected checkpoints, unchanged for all scene groups",
        "prediction_inference_rerun": False,
        "model_sources": [],
        "full_test_regression": [],
    }
    results = []
    for task in ["tower", "tower_top"]:
        for model in ["yolov5n", "yolov8n", "yolov12n"]:
            folder = WORK / "outputs/detector_evaluation_20261020" / f"{task}_{model}"
            meta = load(folder / "metrics.json")
            gt = load(folder / "ground_truth_coco.json")
            pred = load(folder / "predictions_coco.json")
            assert sha(mounted(meta["weights"])) == meta["weight_sha256"]
            for setting, value in [
                ("imgsz", 640),
                ("conf", 0.001),
                ("nms_iou", 0.7),
                ("max_det", 100),
                ("pr_conf", 0.25),
                ("pr_iou", 0.5),
            ]:
                assert meta["settings"][setting] == value
            source = dict(
                task=task,
                model=model,
                predictions=str(folder / "predictions_coco.json"),
                prediction_sha256=sha(folder / "predictions_coco.json"),
                original_gt=str(folder / "ground_truth_coco.json"),
                original_gt_sha256=sha(folder / "ground_truth_coco.json"),
                checkpoint=str(mounted(meta["weights"])),
                checkpoint_sha256=meta["weight_sha256"],
                settings=meta["settings"],
            )
            evidence["model_sources"].append(source)
            full, _ = evaluate(gt, pred, "bbox", 0.25)
            for metric in ["AP_percent", "AP50_percent", "P_percent", "R_percent"]:
                assert_close(
                    full[metric],
                    meta["primary_target_metrics"][metric],
                    (task, model, metric),
                )
            evidence["full_test_regression"].append(
                dict(
                    task=task,
                    model=model,
                    passed=True,
                    **{
                        m: full[m]
                        for m in [
                            "AP_percent",
                            "AP50_percent",
                            "P_percent",
                            "R_percent",
                        ]
                    },
                )
            )
            original_images = {im["id"]: im for im in gt["images"]}
            original_annotations = {a["id"]: a for a in gt["annotations"]}
            for scene in definitions:
                subset = load(ROOT / scene / task / "annotations_coco.json")
                ids = {im["id"] for im in subset["images"]}
                for im in subset["images"]:
                    original = original_images[im["id"]]
                    assert (Path(im["file_name"]).name, im["width"], im["height"]) == (
                        Path(original["file_name"]).name,
                        original["width"],
                        original["height"],
                    )
                assert all(
                    original_annotations[a["id"]] == a for a in subset["annotations"]
                )
                metric, per_image = evaluate(subset, pred, "bbox", 0.25)
                row = dict(
                    scene=scene,
                    scene_name=definitions[scene]["name"],
                    task=task,
                    model=model,
                    **metric,
                )
                results.append(row)
                dump(OUT / "per_image" / f"{scene}_{task}_{model}.json", per_image)
            # Subtypes retain their original memberships; no new sample selection.
            light = [
                r
                for r in scene_index
                if r["task"] == task and r["scene"] == "02_strong_reflection"
            ]
            for subtype in sorted({r["lighting_subtype"] for r in light}):
                ids = {
                    r["original_image_id"]
                    for r in light
                    if r["lighting_subtype"] == subtype
                }
                sub = {
                    **gt,
                    "images": [im for im in gt["images"] if im["id"] in ids],
                    "annotations": [
                        a for a in gt["annotations"] if a["image_id"] in ids
                    ],
                }
                metric, pi = evaluate(sub, pred, "bbox", 0.25)
                results.append(
                    dict(
                        scene="lighting_subtype:" + subtype,
                        scene_name=subtype,
                        task=task,
                        model=model,
                        **metric,
                    )
                )
            print("DETECTOR COMPLETE", task, model, flush=True)

    original_wire_path = Path("I:/datasets/LineDatasetFull/test.json")
    original_wire = load(original_wire_path)
    wire_subset = load(ROOT / "03_vegetation_interference/wire/annotations_coco.json")
    for im in wire_subset["images"]:
        ref = next(x for x in original_wire["images"] if x["id"] == im["id"])
        assert Path(im["file_name"]).name == ref["file_name"] and (
            im["width"],
            im["height"],
        ) == (
            ref["width"],
            ref["height"],
        )
    # Exported dataset annotations may differ in metadata. Compare actual masks and boxes by image/category.
    from pycocotools import mask as mask_utils

    a1 = COCO()
    a1.dataset = deepcopy(original_wire)
    a1.createIndex()
    a2 = COCO()
    a2.dataset = deepcopy(wire_subset)
    a2.createIndex()
    for im in wire_subset["images"]:
        aa = sorted(a1.imgToAnns[im["id"]], key=lambda a: a["id"])
        bb = sorted(a2.imgToAnns[im["id"]], key=lambda a: a["id"])
        assert len(aa) == len(bb)
        for a, b in zip(aa, bb):
            assert a["category_id"] == b["category_id"] and np.allclose(
                a["bbox"], b["bbox"]
            )
            assert np.array_equal(a1.annToMask(a), a2.annToMask(b))
    for model, folder, prefix, report in [
        (
            "yolact",
            (
                Path(__file__).resolve().parents[2]
                / "segmentation/yolact/results/coco_eval"
            ),
            "yolact",
            WORK / "outputs/wire_final_yolact_metrics.json",
        ),
        (
            "maskrcnn",
            (
                Path(__file__).resolve().parents[2]
                / "segmentation/maskrcnn/results/test_wire700_completed"
            ),
            "maskrcnn",
            WORK / "outputs/wire_final_maskrcnn_metrics.json",
        ),
        (
            "rfdetr",
            (
                Path(__file__).resolve().parents[2]
                / "segmentation/rfdetr/results/coco_eval"
            ),
            "rfdetr",
            WORK / "outputs/wire_final_rfdetr_metrics.json",
        ),
    ]:
        meta = load(folder / f"{prefix}_metadata.json")
        predpath = folder / f"{prefix}_segm_predictions.json"
        pred = load(predpath)
        assert (
            meta["image_count"] == 229
            and meta["score_threshold"] == 0.3
            and meta["max_detections"] == 100
        )
        assert meta["annotation_sha256"] == sha(original_wire_path)
        assert sha(mounted(meta["checkpoint"])) == meta["checkpoint_sha256"]
        assert sha(predpath) == meta["prediction_sha256"]["segm"]
        full, _ = evaluate(original_wire, pred, "segm", 0.3)
        expected = load(report)["metrics"]["segm"]
        for metric, field in [
            ("AP_percent", "AP"),
            ("AP50_percent", "AP50"),
            ("AP75_percent", "AP75"),
        ]:
            assert_close(full[metric], expected[field], (model, metric))
        evidence["full_test_regression"].append(
            dict(
                task="wire",
                model=model,
                passed=True,
                **{m: full[m] for m in ["AP_percent", "AP50_percent", "AP75_percent"]},
            )
        )
        evidence["model_sources"].append(
            dict(
                task="wire",
                model=model,
                predictions=str(predpath),
                prediction_sha256=sha(predpath),
                checkpoint=str(mounted(meta["checkpoint"])),
                checkpoint_sha256=meta["checkpoint_sha256"],
                annotation_sha256=meta["annotation_sha256"],
                score_threshold=meta["score_threshold"],
                resolution=meta.get("resolution"),
            )
        )
        if model == "rfdetr":
            # Share the same local predictions as the matched-data table, rather than
            # introducing a second numerical realization of this nine-image evaluation.
            matched_folder = (
                Path(__file__).resolve().parents[2]
                / "segmentation/rfdetr/results/revision5_matched_completed_20261102"
            )
            local_path = matched_folder / "TTPLA_plus_local_predictions.json"
            local = load(local_path)
            expected_local = load(matched_folder / "TTPLA_plus_local_metrics.json")
            assert local["metadata"]["checkpoint_sha256"] == meta["checkpoint_sha256"]
            assert (
                local["metadata"]["score_threshold"] == 0.3
                and local["metadata"]["inference_resolution"] == 648
            )
            assert local["metadata"]["max_cable_detections"] == 100
            assert set(local["metadata"]["image_ids"]) == {
                im["id"] for im in wire_subset["images"]
            }
            assert sha(local_path) == expected_local["prediction_sha256"]
            pred = local["predictions"]["segm"]
            evidence["model_sources"][-1]["local_scene_predictions"] = str(local_path)
            evidence["model_sources"][-1]["local_scene_predictions_sha256"] = sha(
                local_path
            )
            evidence["model_sources"][-1]["local_scene_device"] = local["metadata"][
                "device"
            ]
            evidence["model_sources"][-1]["local_scene_source_reason"] = (
                "Identical local predictions shared with the existing matched-data evaluation table; checkpoint and task thresholds unchanged."
            )
        metric, pi = evaluate(wire_subset, pred, "segm", 0.3)
        if model == "rfdetr":
            for metric_name, field in [
                ("AP_percent", "AP"),
                ("AP50_percent", "AP50"),
                ("AP75_percent", "AP75"),
            ]:
                assert_close(
                    metric[metric_name],
                    expected_local["metrics"]["segm"][field],
                    ("rfdetr_local_consistency", metric_name),
                )
        results.append(
            dict(
                scene="03_vegetation_interference",
                scene_name=definitions["03_vegetation_interference"]["name"],
                task="wire",
                model=model,
                **metric,
            )
        )
        dump(OUT / "per_image" / f"vegetation_wire_{model}.json", pi)
        print("SEGMENTER COMPLETE", model, flush=True)

    direction_path = LAB / "results/test_20261102_162346_510106/per_region.json"
    regions = load(direction_path)
    keys = {
        r["key"].removeprefix("angle_") for r in scene_index if r["task"] == "angle"
    }
    # Python 3.9 supports removeprefix; the original annotation keys stay unchanged.
    assert len(keys) == 10
    dr = []
    for method in ["original", "probabilistic"]:
        selected = [
            r
            for r in regions
            if r["method"] == method
            and r["mask_source"] == "pred"
            and r["variant"] == "default"
        ]
        assert len(selected) == 56
        for group, rs in [
            ("vegetation", [r for r in selected if r["key"] in keys]),
            ("other_daytime", [r for r in selected if r["key"] not in keys]),
            ("all", selected),
        ]:
            metric = summarize(rs)
            dr.append(dict(method=method, scene=group, **metric))
            dump(OUT / "per_region" / f"{group}_{method}.json", rs)
    evidence["direction_source"] = {
        "per_region": str(direction_path),
        "sha256": sha(direction_path),
        "frozen_protocol_sha256": sha(LAB / "frozen_protocol.json"),
        "selected_keys": sorted(keys),
        "existing_methods_and_settings_unchanged": True,
    }
    for method in ["original", "probabilistic"]:
        full = next(x for x in dr if x["method"] == method and x["scene"] == "all")
        original_summary = next(
            x
            for x in load(direction_path.parent / "summary.json")
            if x["method"] == method
            and x["mask_source"] == "pred"
            and x["variant"] == "default"
            and x["group"] == "all"
        )
        for k in [
            "axis_MAE_deg_matched",
            "axis_P95_deg_matched",
            "direction_precision_at_5deg",
            "direction_recall_at_5deg",
        ]:
            assert_close(full[k], original_summary[k], (method, k))
    dump(OUT / "recognition_metrics.json", results)
    dump(OUT / "direction_metrics.json", dr)
    dump(OUT / "evaluation_evidence.json", evidence)
    columns = [
        "scene",
        "task",
        "model",
        "images",
        "target_instances",
        "positive_images",
        "AP_percent",
        "AP50_percent",
        "AP75_percent",
        "P_percent",
        "R_percent",
        "success_percent",
        "success_images",
        "TP",
        "FP",
        "FN",
        "background_images_with_false_positive",
    ]
    with (OUT / "recognition_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    with (OUT / "direction_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(dr[0]))
        w.writeheader()
        w.writerows(dr)
    print(
        "ALL COMPLETE",
        len(results),
        "recognition rows;",
        len(dr),
        "direction rows",
        flush=True,
    )
    print(
        json.dumps(
            [r for r in results if not r["scene"].startswith("lighting_subtype:")],
            ensure_ascii=True,
            indent=2,
        )
    )
    print(json.dumps(dr, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
