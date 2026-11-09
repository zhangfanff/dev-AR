"""Evaluate TTPLA-only and mixed-data checkpoints on local wire images."""

from pathlib import Path
import argparse, copy, csv, hashlib, importlib.metadata, json, sys
import numpy as np
from PIL import Image
import torch
from pycocotools import mask as mask_api
from pycocotools.coco import COCO

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from coco_evaluation import evaluate_predictions


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(
        json.dumps(d, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def audit(out):
    baseline = PROJECT / "model_ttpla_cable_matched648"
    mixed = PROJECT / "model_new648"
    protocol = read(baseline / "matched_training_protocol.json")
    manifests = read(
        PROJECT / "dataset_ttpla_cable_matched648/matched_dataset_manifest.json"
    )
    assert protocol["all_saved_learning_settings_match"]
    args = []
    hashes = {}
    training = []
    for name, folder in [("TTPLA_only", baseline), ("TTPLA_plus_local", mixed)]:
        p = folder / "checkpoint_best_total.pth"
        c = torch.load(p, map_location="cpu", weights_only=False)
        assert c["model"]["class_embed.weight"].shape[0] == 1
        args.append(vars(c["args"]))
        hashes[name] = sha(p)
        del c
        logs = [
            json.loads(s)
            for s in (folder / "log.txt").read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
        assert [r["epoch"] for r in logs] == list(range(10))
        candidates = [
            dict(variant=variant, epoch=r["epoch"], mask_AP=r[key][0] * 100)
            for r in logs
            for key, variant in [
                ("test_coco_eval_masks", "regular"),
                ("ema_test_coco_eval_masks", "EMA"),
            ]
        ]
        best = max(candidates, key=lambda r: r["mask_AP"])
        training.append(
            dict(
                name=name,
                checkpoint=str(p),
                checkpoint_sha256=hashes[name],
                training_epochs=10,
                validation_best=best,
            )
        )
    allowed = {"dataset_dir", "output_dir", "pretrain_weights", "run_test"}
    differences = {
        k: {"baseline": args[0].get(k), "mixed": args[1].get(k)}
        for k in sorted(set(args[0]) | set(args[1]))
        if args[0].get(k) != args[1].get(k)
    }
    assert not set(differences) - allowed, differences
    for a in args:
        assert {
            k: a[k]
            for k in [
                "resolution",
                "num_classes",
                "batch_size",
                "grad_accum_steps",
                "epochs",
                "seed",
            ]
        } == dict(
            resolution=648,
            num_classes=1,
            batch_size=1,
            grad_accum_steps=4,
            epochs=10,
            seed=42,
        )
        initial = Path(a["pretrain_weights"])
        initial = initial if initial.is_absolute() else PROJECT / initial
        assert sha(initial) == protocol["initial_weights_sha256"]
    assert hashes["TTPLA_plus_local"] == protocol["reference_checkpoint_sha256"]
    data = PROJECT / "dataset_new"
    source = PROJECT / "dataset_ttpla_cable_matched648"
    assert sha(data / "valid/_annotations.coco.json") == sha(
        source / "valid/_annotations.coco.json"
    )
    train = read(data / "train/_annotations.coco.json")
    source_train = read(source / "train/_annotations.coco.json")
    assert len(train["images"]) == 961 and len(source_train["images"]) == 905
    assert (
        len(source_train["categories"]) == 1
        and source_train["categories"][0]["name"] == "cable"
    )
    public_names = {
        i["file_name"] for i in train["images"] if not i["file_name"].startswith("DJI")
    }
    assert public_names == {i["file_name"] for i in source_train["images"]}
    ids = {i["id"] for i in train["images"] if i["file_name"] in public_names}

    # Exporters may renumber annotation IDs. Compare actual labels by filename,
    # keeping polygons/boxes/classes intact rather than treating IDs as geometry.
    def canonical(doc, selected_ids=None):
        names = {i["id"]: i["file_name"] for i in doc["images"]}
        return sorted(
            (
                names[a["image_id"]],
                json.dumps(
                    {k: v for k, v in a.items() if k not in ("id", "image_id")},
                    sort_keys=True,
                ),
            )
            for a in doc["annotations"]
            if selected_ids is None or a["image_id"] in selected_ids
        )

    assert canonical(train, ids) == canonical(source_train)
    for split in ["train", "valid"]:
        assert (
            sha(source / split / "_annotations.coco.json")
            == manifests["splits"][split]["annotation_sha256"]
        )
        for name, h in manifests["image_sha256"][split].items():
            assert sha(source / split / name) == h
            assert sha(data / split / name) == h
    full = read(data / "test/_annotations.coco.json")
    local = [i for i in full["images"] if i["file_name"].startswith("DJI")]
    keep = {i["id"] for i in local}
    sub = dict(
        copy.deepcopy(full),
        images=local,
        annotations=[a for a in full["annotations"] if a["image_id"] in keep],
    )
    sub.setdefault("info", {})
    assert len(local) == 9 and len(sub["annotations"]) == 65
    htest = {i["file_name"]: sha(data / "test" / i["file_name"]) for i in local}
    overlap = {}
    for split in ["train", "valid"]:
        doc = read(data / split / "_annotations.coco.json")
        hs = {sha(data / split / i["file_name"]) for i in doc["images"]}
        overlap[split] = len(set(htest.values()) & hs)
        assert not overlap[split]
    annotation = out / "local_test_annotations.json"
    write(annotation, sub)
    report = dict(
        training_conditions=training,
        saved_learning_settings_match=True,
        permitted_nonlearning_differences=differences,
        shared_validation_images=125,
        validation_includes_local_images=16,
        initial_weights_sha256=protocol["initial_weights_sha256"],
        training_image_counts={"TTPLA_only": 905, "TTPLA_plus_local": 961},
        test_images=9,
        test_instances=65,
        test_image_sha256=htest,
        test_annotation_sha256=sha(annotation),
        exact_image_overlap_with_training_or_validation=overlap,
        scope="Single-seed comparison with matching saved hyperparameters and ten epochs; data composition changes sample/update exposure. Not route-level cross-validation or a zero-target-label selection claim.",
    )
    write(out / "matched_audit.json", report)
    print(
        "AUDIT PASSED: identical saved learning settings, common validation, nine held-out local images.",
        flush=True,
    )
    return sub, annotation, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "results/revision5_matched_completed_20261102",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    sub, annotation, aud = audit(args.out)
    from rfdetr import RFDETRSegPreview

    results = []
    for condition in aud["training_conditions"]:
        name = condition["name"]
        checkpoint = Path(condition["checkpoint"])
        metadata = dict(
            **condition,
            annotation_sha256=sha(annotation),
            image_ids=sorted(i["id"] for i in sub["images"]),
            device=args.device,
            inference_resolution=648,
            num_select=200,
            score_threshold=0.3,
            max_cable_detections=100,
            torch=torch.__version__,
            rfdetr=importlib.metadata.version("rfdetr"),
        )
        cache = args.out / (name + "_predictions.json")
        if cache.exists():
            previous = read(cache)
            assert previous["metadata"] == metadata
            predictions = previous["predictions"]
            print("Reusing audited predictions:", name, flush=True)
        else:
            model = RFDETRSegPreview(
                pretrain_weights=str(checkpoint),
                resolution=648,
                num_select=200,
                device=args.device,
            )
            assert model.model.class_names == ["cable"]
            predictions = {"bbox": [], "segm": []}
            for i, info in enumerate(
                sorted(sub["images"], key=lambda item: item["id"])
            ):
                with Image.open(
                    PROJECT / "dataset_new/test" / info["file_name"]
                ) as image:
                    pixels = np.array(image.convert("RGB"))
                with torch.inference_mode():
                    pred = model.predict(pixels, threshold=0.3)
                selected = np.flatnonzero(pred.class_id == 0)
                selected = selected[
                    np.argsort(-pred.confidence[selected], kind="stable")[:100]
                ]
                for j in selected:
                    common = {
                        "image_id": info["id"],
                        "category_id": 0,
                        "score": float(pred.confidence[j]),
                    }
                    x1, y1, x2, y2 = map(float, pred.xyxy[j])
                    predictions["bbox"].append(
                        dict(common, bbox=[x1, y1, x2 - x1, y2 - y1])
                    )
                    mask = np.asarray(pred.mask[j])
                    assert mask.shape == (info["height"], info["width"])
                    rle = mask_api.encode(
                        np.asfortranarray((mask > 0).astype(np.uint8))
                    )
                    rle["counts"] = rle["counts"].decode("ascii")
                    predictions["segm"].append(dict(common, segmentation=rle))
                print(
                    name,
                    i + 1,
                    len(sub["images"]),
                    "predictions",
                    len(selected),
                    flush=True,
                )
            write(cache, dict(metadata=metadata, predictions=predictions))
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        gt = COCO(str(annotation))
        metrics = evaluate_predictions(gt, predictions, metadata["image_ids"], 100)
        result = dict(
            **metadata,
            metrics=metrics,
            prediction_sha256=sha(cache),
            test_images=9,
            test_instances=65,
        )
        write(args.out / (name + "_metrics.json"), result)
        results.append(result)
        print("METRICS", name, json.dumps(metrics["segm"]), flush=True)
    write(args.out / "comparison.json", results)
    rows = [
        dict(
            training_data=r["name"],
            mask_AP=r["metrics"]["segm"]["AP"],
            mask_AP50=r["metrics"]["segm"]["AP50"],
            mask_AP75=r["metrics"]["segm"]["AP75"],
            box_AP=r["metrics"]["bbox"]["AP"],
        )
        for r in results
    ]
    with (args.out / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("MATCHED COMPARISON COMPLETE", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
