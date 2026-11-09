"""Train a TTPLA cable baseline using the reference checkpoint configuration."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil

PROJECT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = PROJECT / "model_new648/checkpoint_best_total.pth"
DEFAULT_DATASET = PROJECT / "dataset_ttpla_cable_matched648"
DEFAULT_OUTPUT = PROJECT / "model_ttpla_cable_matched648"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def canonical_annotations(data):
    names = {i["id"]: i["file_name"] for i in data["images"]}
    return sorted(
        (
            names[a["image_id"]],
            json.dumps(
                {k: v for k, v in a.items() if k not in ("id", "image_id")},
                sort_keys=True,
            ),
        )
        for a in data["annotations"]
    )


def cable_only(data):
    cable = [c for c in data["categories"] if c["name"] == "cable"]
    if len(cable) != 1:
        raise ValueError("Exactly one cable category is required.")
    category_id = cable[0]["id"]
    out = copy.deepcopy(data)
    out["categories"] = [{"supercategory": "cable", "id": 0, "name": "cable"}]
    out["annotations"] = [
        dict(a, category_id=0)
        for a in out["annotations"]
        if a["category_id"] == category_id
    ]
    return out


def ttpla_part_of_mixed(data):
    out = copy.deepcopy(data)
    out["images"] = [i for i in data["images"] if not i["file_name"].startswith("DJI")]
    ids = {i["id"] for i in out["images"]}
    out["annotations"] = [a for a in data["annotations"] if a["image_id"] in ids]
    return out


def prepare_dataset(destination):
    source = PROJECT / "dataset_ttpla"
    mixed = PROJECT / "dataset_new"
    paths = {
        "train": source / "train/_annotations.coco.json",
        "valid": mixed / "valid/_annotations.coco.json",
        "test": source / "test/_annotations.coco.json",
        "mixed_train": mixed / "train/_annotations.coco.json",
        "mixed_test": mixed / "test/_annotations.coco.json",
    }
    input_hashes = {k: digest(p) for k, p in paths.items()}
    manifest_path = destination / "matched_dataset_manifest.json"
    if (
        destination.exists()
        and any(destination.iterdir())
        and not manifest_path.exists()
    ):
        raise RuntimeError(
            "Dataset destination is not an audited generated directory: "
            + str(destination)
        )
    if manifest_path.exists():
        prior = read_json(manifest_path)
        if prior["source_annotation_sha256"] != input_hashes:
            raise RuntimeError("Source annotations changed; use a fresh --dataset-dir.")

    data = {
        "train": cable_only(read_json(paths["train"])),
        "valid": read_json(paths["valid"]),
        "test": cable_only(read_json(paths["test"])),
    }
    common_train = ttpla_part_of_mixed(read_json(paths["mixed_train"]))
    if canonical_annotations(data["train"]) != canonical_annotations(common_train):
        raise ValueError(
            "TTPLA cable annotations differ from the mixed model training subset."
        )
    assert {i["file_name"] for i in data["train"]["images"]} == {
        i["file_name"] for i in common_train["images"]
    }
    assert data["valid"]["categories"] == [
        {"supercategory": "cable", "id": 0, "name": "cable"}
    ]
    assert not any(
        i["file_name"].startswith("DJI")
        for s in ("train", "test")
        for i in data[s]["images"]
    )
    expected = {"train": (905, 7282), "valid": (125, 954), "test": (220, 1968)}
    for split, counts in expected.items():
        assert (
            len(data[split]["images"]),
            len(data[split]["annotations"]),
        ) == counts, (
            split,
            counts,
        )

    source_roots = {
        "train": source / "train",
        "valid": mixed / "valid",
        "test": source / "test",
    }
    image_hashes = {}
    for split in ("train", "valid", "test"):
        folder = destination / split
        folder.mkdir(parents=True, exist_ok=True)
        image_hashes[split] = {}
        for image in data[split]["images"]:
            name = image["file_name"]
            if Path(name).name != name:
                raise ValueError("Expected a flat exported image name: " + name)
            original = source_roots[split] / name
            h = digest(original)
            target = folder / name
            if target.exists():
                if digest(target) != h:
                    raise RuntimeError(
                        "Generated image differs from its source: " + str(target)
                    )
            else:
                shutil.copy2(original, target)
            image_hashes[split][name] = h
            if split == "train" and h != digest(mixed / "train" / name):
                raise ValueError(
                    "TTPLA image pixels differ from the mixed training image: " + name
                )
        annotation = folder / "_annotations.coco.json"
        if split == "valid":
            # Byte-identical common validation annotations, including local validation images.
            if annotation.exists() and digest(annotation) != input_hashes["valid"]:
                raise RuntimeError(
                    "Existing common validation annotations were changed."
                )
            shutil.copy2(paths["valid"], annotation)
        else:
            if annotation.exists() and read_json(annotation) != data[split]:
                raise RuntimeError(
                    "Existing generated annotations were changed: " + str(annotation)
                )
            write_json(annotation, data[split])

    local_test = [
        i
        for i in read_json(paths["mixed_test"])["images"]
        if i["file_name"].startswith("DJI")
    ]
    assert len(local_test) == 9
    test_hashes = {digest(mixed / "test" / i["file_name"]) for i in local_test}
    overlap = {
        s: len(test_hashes & set(image_hashes[s].values())) for s in ("train", "valid")
    }
    assert not any(overlap.values()), overlap
    assert not set(image_hashes["train"].values()) & set(image_hashes["valid"].values())
    manifest = {
        "training_condition": "TTPLA cable images only; local validation images used for common checkpoint selection, never for gradient updates.",
        "source_annotation_sha256": input_hashes,
        "splits": {},
        "image_sha256": image_hashes,
        "local_test_image_count": 9,
        "local_test_overlap_sha256": overlap,
        "source_images_copied_unchanged": True,
        "original_datasets_modified": False,
    }
    for split in data:
        annotated = {a["image_id"] for a in data[split]["annotations"]}
        manifest["splits"][split] = {
            "images": len(data[split]["images"]),
            "instances": len(data[split]["annotations"]),
            "empty_images": sum(
                i["id"] not in annotated for i in data[split]["images"]
            ),
            "local_images": sum(
                i["file_name"].startswith("DJI") for i in data[split]["images"]
            ),
            "annotation_sha256": digest(destination / split / "_annotations.coco.json"),
        }
    write_json(manifest_path, manifest)
    print("DATASET", json.dumps(manifest["splits"]), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and audit data/config, without loading a model or training.",
    )
    p.add_argument(
        "--check-model",
        action="store_true",
        help="Verify effective settings and one forward pass; no optimizer updates.",
    )
    args = p.parse_args()
    os.chdir(PROJECT)
    import numpy as np
    import torch
    from rfdetr import RFDETRSegPreview
    from rfdetr.config import RFDETRSegPreviewConfig
    from rfdetr.main import populate_args

    ck = torch.load(args.reference, map_location="cpu", weights_only=False)
    reference = copy.deepcopy(vars(ck["args"]))
    assert ck["model"]["class_embed.weight"].shape[0] == 1
    del ck
    required = {
        "resolution": 648,
        "num_classes": 1,
        "batch_size": 1,
        "grad_accum_steps": 4,
        "epochs": 10,
        "seed": 42,
        "lr": 1e-4,
        "num_workers": 0,
        "world_size": 1,
        "segmentation_head": True,
        "multi_scale": True,
        "expanded_scales": True,
        "resume": "",
        "start_epoch": 0,
        "class_names": ["cable"],
    }
    for key, value in required.items():
        if reference.get(key) != value:
            raise ValueError(
                "Unexpected reference setting: " + key + " = " + str(reference.get(key))
            )
    initial = PROJECT / reference["pretrain_weights"]
    if not initial.is_file() or initial.name != "rf-detr-seg-preview.pt":
        raise FileNotFoundError(
            "The original generic pretrained initialization is required: "
            + str(initial)
        )
    args.output_dir = args.output_dir.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    if any(args.output_dir.glob("checkpoint*.pth")):
        raise RuntimeError(
            "This output already has trained checkpoints; use a new --output-dir to avoid overwriting."
        )
    manifest = prepare_dataset(args.dataset_dir)
    kwargs = copy.deepcopy(reference)
    kwargs.update(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        pretrain_weights=str(initial.resolve()),
        run_test=False,
    )
    # run_test only controls optional post-training testing; it does not change learning.
    permitted = {"dataset_dir", "output_dir", "pretrain_weights", "run_test"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "reference_checkpoint": str(args.reference.resolve()),
        "reference_checkpoint_sha256": digest(args.reference),
        "initial_weights": str(initial),
        "initial_weights_sha256": digest(initial),
        "rfdetr_version": importlib.metadata.version("rfdetr"),
        "torch_version": torch.__version__,
        "reference_args": reference,
        "requested_args": kwargs,
        "dataset_summary": manifest["splits"],
        "nominal_resolution": 648,
        "micro_batch": 1,
        "gradient_accumulation": 4,
        "effective_batch": 4,
        "shared_validation": True,
        "automatic_test_disabled": True,
        "note": "Same saved learning and augmentation settings. Equal epochs do not imply equal numbers of training samples/optimizer steps.",
    }
    write_json(args.output_dir / "matched_training_protocol.json", protocol)
    print(
        "MATCHED CONFIG: base resolution 648, cable=1, batch=1, accumulation=4, effective batch=4, epochs=10.",
        flush=True,
    )
    print(
        "Training starts from generic pretrained weights, not the mixed-data trained checkpoint.",
        flush=True,
    )
    if args.prepare_only:
        print("PREPARE ONLY: no model or optimizer was started.", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Use the original CUDA-enabled RF-DETR environment.")
    seed = reference["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_fields = getattr(RFDETRSegPreviewConfig, "model_fields", None)
    if model_fields is None:
        model_fields = RFDETRSegPreviewConfig.__fields__
    model_config = {k: v for k, v in kwargs.items() if k in model_fields}
    model = RFDETRSegPreview(**model_config)
    # Generic pretraining may restore a multi-class head. Explicitly retain its first
    # cable row, using the same library operation used by the original training API.
    model.model.reinitialize_detection_head(1)
    model.model.class_names = ["cable"]
    assert model.model.model.class_embed.weight.shape[0] == 1
    original_train = model.model.train

    def checked_training(**effective):
        callbacks = effective.pop("callbacks")
        actual = vars(populate_args(**effective))
        changes = {
            k: {"reference": v, "actual": actual.get(k)}
            for k, v in reference.items()
            if k not in permitted and actual.get(k) != v
        }
        if changes:
            raise RuntimeError(
                "Refusing unmatched training settings: "
                + json.dumps(changes, default=str)
            )
        protocol["effective_args"] = actual
        protocol["all_saved_learning_settings_match"] = True
        protocol["permitted_changes"] = {
            k: {"reference": reference.get(k), "actual": actual.get(k)}
            for k in permitted
            if reference.get(k) != actual.get(k)
        }
        write_json(args.output_dir / "matched_training_protocol.json", protocol)
        if args.check_model:
            from rfdetr.datasets import build_dataset
            from rfdetr.util.misc import nested_tensor_from_tensor_list

            dataset = build_dataset("val", argparse.Namespace(**actual), resolution=648)
            image, target = dataset[0]
            assert len(dataset) == 125 and list(image.shape) == [3, 648, 648]
            model.model.model.eval()
            batch = nested_tensor_from_tensor_list([image]).to("cuda")
            with (
                torch.inference_mode(),
                torch.autocast("cuda", enabled=actual["amp"], dtype=torch.bfloat16),
            ):
                result = model.model.model(batch)
            assert result["pred_logits"].shape[-1] == 1 and "pred_masks" in result
            protocol["preflight"] = {
                "mode": "forward only; no optimizer or updates",
                "validation_count": len(dataset),
                "input_shape": list(image.shape),
                "logit_shape": list(result["pred_logits"].shape),
                "mask_shape": list(result["pred_masks"].shape),
                "success": True,
            }
            write_json(
                args.output_dir / "matched_preflight.json", protocol["preflight"]
            )
            print("PREFLIGHT PASSED", json.dumps(protocol["preflight"]), flush=True)
            return
        print(
            "TRAINING START: checkpoint selection uses common-validation mask AP only.",
            flush=True,
        )
        return original_train(callbacks=callbacks, **effective)

    model.model.train = checked_training
    model.train(**kwargs)
    if not args.check_model:
        print(
            "Training complete. Baseline checkpoint:",
            args.output_dir / "checkpoint_best_total.pth",
            flush=True,
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
