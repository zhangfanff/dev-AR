"""Prepare nadir-view local images and preserve existing annotations."""

from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import collections
import re
import shutil
from PIL import Image, ImageOps
from wirelab.io import ROOT, read, write, sha

PROJECT = ROOT.parent / "segmentation/rfdetr"
RAW = Path("I:/datasets/LineDataset")
EXTERNAL = [
    Path(p)
    for p in (
        "I:/配网无人机相关/2026-10",
        "I:/配网无人机相关/素材(1)",
        "I:/配网无人机相关/new_towertop",
        "I:/datasets/sucai0701",
    )
]
PITCH_TOLERANCE = 5.0


def camera_metadata(path):
    with Path(path).open("rb") as stream:
        header = stream.read(1024 * 1024)
    return {
        k.decode(): float(v)
        for k, v in re.findall(
            rb'(GimbalPitchDegree|GimbalRollDegree|GimbalYawDegree|FlightRollDegree|FlightPitchDegree)="([-+\d.]+)"',
            header,
        )
    }


def nadir_status(meta):
    pitch = meta.get("GimbalPitchDegree")
    roll = meta.get("GimbalRollDegree")
    if pitch is None or roll is None:
        return "missing_camera_metadata"
    if abs(pitch + 90) > PITCH_TOLERANCE or abs(roll) > PITCH_TOLERANCE:
        return "non_nadir_camera_pose"
    return "eligible"


def field_group(path):
    uuid = re.search(r"_V-([A-Fa-f0-9-]{36})-", path.stem)
    if uuid:
        return (
            re.search(r"DJI_(\d{8})", path.name).group(1) + "/" + uuid.group(1).upper()
        )
    return path.stem


def snapshot(root):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = root / "archive" / ("before_local_nadir_" + stamp + ".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for folder in ("dataset", "annotations"):
            for p in (root / folder).rglob("*"):
                if p.is_file():
                    archive.write(p, p.relative_to(root).as_posix())
        p = root / "frozen_protocol.json"
        if p.exists():
            archive.write(p, p.name)
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Backup validation failed.")
    return target


def build(root):
    out = root / "dataset"
    old = read(out / "manifest.json") if (out / "manifest.json").exists() else {}
    old_cache = {}
    for p in (out / "predictions").glob("*.json"):
        d = read(p)
        if d.get("metadata", {}).get("checkpoint_sha256") not in (
            None,
            old.get("prediction_checkpoint_sha256"),
        ):
            raise RuntimeError("Prediction model mismatch: " + str(p))
        old_cache.update(d.get("records", {}))
    selected = []
    exclusions = []
    documents = {}
    originals = {}
    all_names = set()
    for split in ("train", "valid", "test"):
        source = PROJECT / "dataset_new" / split / "_annotations.coco.json"
        doc = read(source)
        documents[split] = doc
        for im in doc["images"]:
            name = im["file_name"]
            if not name.startswith("DJI"):
                continue
            all_names.add(name)
            raw = RAW / name
            meta = camera_metadata(raw)
            reason = nadir_status(meta)
            item = dict(
                key=f"{split}_{im['id']:06d}",
                image_id=im["id"],
                file_name=name,
                width=im["width"],
                height=im["height"],
                original_split=split,
                role="test" if split == "test" else "dev",
                source="local",
                has_gt_mask=True,
                segmentation_training_seen=split == "train",
                source_path=str(raw),
                camera_metadata=meta,
                viewpoint_screen="dji_xmp_pitch_and_roll_within_5deg_of_nadir",
                selection_reason="all_local_images_passing_camera_pose_filter",
            )
            if reason != "eligible":
                exclusions.append(
                    dict(
                        key=item["key"],
                        file_name=name,
                        source_path=str(raw),
                        reason=reason,
                        camera_metadata=meta,
                    )
                )
                continue
            selected.append(item)
            originals[item["key"]] = source.parent / name
    candidates = []
    for base in EXTERNAL:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in (
                ".jpg",
                ".jpeg",
                ".png",
            ) or not p.name.startswith("DJI"):
                continue
            meta = camera_metadata(p)
            reason = nadir_status(meta)
            if p.name in all_names:
                reason = "already_in_segmentation_dataset"
            if reason != "eligible":
                exclusions.append(
                    dict(
                        file_name=p.name,
                        source_path=str(p),
                        reason=reason,
                        camera_metadata=meta,
                    )
                )
                continue
            candidates.append((p, meta))
    names = set()
    hashes = set()
    groups = set()
    for p, meta in sorted(candidates, key=lambda pair: (pair[0].name, str(pair[0]))):
        digest = sha(p)
        group = field_group(p)
        reason = (
            "duplicate_filename_or_bytes"
            if p.name in names or digest in hashes
            else "repeated_tower_within_recording_session"
            if group in groups
            else None
        )
        if reason:
            exclusions.append(
                dict(
                    file_name=p.name,
                    source_path=str(p),
                    reason=reason,
                    camera_metadata=meta,
                )
            )
            continue
        names.add(p.name)
        hashes.add(digest)
        groups.add(group)
        key = "local_" + re.search(r"DJI_(\d{14}_\d{4})", p.name).group(1)
        with Image.open(p) as im:
            size = list(im.size)
        item = dict(
            key=key,
            image_id=100000 + len(selected),
            file_name=p.name,
            width=700,
            height=700,
            original_split="external",
            role="test",
            source="local",
            has_gt_mask=False,
            segmentation_training_seen=False,
            source_path=str(p),
            source_sha256=digest,
            original_size=size,
            recording_group=group,
            camera_metadata=meta,
            viewpoint_screen="dji_xmp_pitch_and_roll_within_5deg_of_nadir",
            selection_reason="local_field_nadir_first_capture_per_recorded_tower",
            image_transform="EXIF transpose then resize to 700x700; no crop, matching existing dataset geometry",
        )
        selected.append(item)
        originals[key] = p
    if not selected:
        raise RuntimeError("No eligible local images; active data unchanged.")
    if len({i["key"] for i in selected}) != len(selected):
        raise RuntimeError("Duplicate stable key.")
    previous = {i["key"]: i for i in old.get("images", [])}
    for i in selected:
        prev = previous.get(i["key"])
        if prev and sha(root / prev["image_path"]) != prev["image_sha256"]:
            raise RuntimeError("Existing image changed: " + i["key"])
    backup = snapshot(root) if old else None
    (out / "images").mkdir(parents=True, exist_ok=True)
    for i in selected:
        target = out / "images" / (i["key"] + ".jpg")
        prev = previous.get(i["key"])
        if i["original_split"] == "external":
            if not (
                prev
                and prev.get("source_sha256") == i["source_sha256"]
                and target.exists()
            ):
                with Image.open(originals[i["key"]]) as im:
                    ImageOps.exif_transpose(im).convert("RGB").resize(
                        (700, 700), Image.Resampling.LANCZOS
                    ).save(target, quality=95)
        else:
            shutil.copy2(originals[i["key"]], target)
        i["image_path"] = target.relative_to(root).as_posix()
        i["image_sha256"] = sha(target)
        label = root / "annotations" / (i["key"] + ".json")
        if label.exists() and read(label).get("image_sha256") != i["image_sha256"]:
            raise RuntimeError(
                "Annotation image mismatch; label preserved: " + i["key"]
            )
    if len({i["image_sha256"] for i in selected}) != len(selected):
        raise RuntimeError("Duplicate selected bytes.")
    active = {i["key"] for i in selected}
    archived = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for p in (root / "annotations").glob("*.json"):
        if p.stem not in active:
            target = root / "archive" / ("excluded_labels_" + stamp) / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            assert p.resolve().parent == (root / "annotations").resolve()
            assert (root / "archive").resolve() in target.resolve().parents
            shutil.move(str(p), str(target))
            archived.append(str(target))
    for split, doc in documents.items():
        ids = {i["image_id"] for i in selected if i["original_split"] == split}
        write(
            out / f"coco_{split}.json",
            dict(
                doc,
                images=[i for i in doc["images"] if i["id"] in ids],
                annotations=[a for a in doc["annotations"] if a["image_id"] in ids],
            ),
        )
    records = {k: v for k, v in old_cache.items() if k in active}
    checkpoint = old.get("prediction_checkpoint_sha256") or sha(
        PROJECT / "model_new648/checkpoint_best_total.pth"
    )
    for p in (out / "predictions").glob("*.json"):
        assert p.resolve().parent == (out / "predictions").resolve()
        p.unlink()  # Only generated caches; all previous records are in the backup ZIP.
    write(
        out / "predictions/migrated_local.json",
        dict(
            metadata={
                "checkpoint_sha256": checkpoint,
                "provenance": "Unchanged local records from prior caches; original CPU/GPU metadata retained in archive.",
            },
            records=records,
        ),
    )
    frozen = root / "frozen_protocol.json"
    if frozen.exists():
        frozen.unlink()  # Dataset revision invalidates freeze; archived above.
    roles = collections.Counter(i["role"] for i in selected)
    m = dict(
        schema="wire_direction_dataset_local_nadir_v2",
        revision=datetime.now().isoformat(),
        local_only=True,
        selection_uses_prediction_quality=False,
        pitch_tolerance_deg=PITCH_TOLERANCE,
        dev_images=roles["dev"],
        test_images=roles["test"],
        images=selected,
        prediction_checkpoint_sha256=checkpoint,
        input_annotation_hashes={
            s: sha(PROJECT / "dataset_new" / s / "_annotations.coco.json")
            for s in documents
        },
        source_split_note="Original splits retained. Training images are development only; external photos are absent by filename from all segmentation splits. Route independence is not established.",
        external_gt_note="External photos have no segmentation GT. Human reference lines define direction truth. Use predicted masks for the primary comparison.",
        annotation_archive=archived,
        backup=str(backup) if backup else None,
    )
    write(
        out / "viewpoint_audit.json",
        dict(
            criterion="abs(gimbal_pitch+90)<=5 AND abs(gimbal_roll)<=5; both required",
            selected_count=len(selected),
            exclusions=exclusions,
            note="DJI XMP screen; annotator can reject oblique/uncertain frames. Wire slant in the image is not camera tilt.",
        ),
    )
    write(out / "manifest.json", m)
    used = {(root / i["image_path"]).resolve() for i in selected}
    for p in (out / "images").iterdir():
        if p.is_file() and p.resolve() not in used:
            assert p.resolve().parent == (out / "images").resolve()
            p.unlink()  # Unused generated copies only, originals and ZIP intact.
    print("Local-only nadir images:", len(selected), dict(roles))
    print(
        "Available GT masks:",
        sum(i["has_gt_mask"] for i in selected),
        "cached predictions:",
        len(records),
    )
    print("Images:", out / "images")
    print("Labels:", root / "annotations")
    print("Excluded labels preserved:", archived)
    print("Backup:", backup)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--refresh-selection", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    root = args.root.resolve()
    if (root / "dataset/manifest.json").exists() and not (
        args.rebuild or args.refresh_selection
    ):
        d = read(root / "dataset/manifest.json")
        if not d.get("local_only"):
            raise RuntimeError(
                "Old mixed selection. Use --rebuild to archive and migrate."
            )
        for i in d["images"]:
            if (
                i["source"] != "local"
                or sha(root / i["image_path"]) != i["image_sha256"]
            ):
                raise RuntimeError("Image integrity: " + i["key"])
        print("Verified local-only images:", len(d["images"]))
        return
    build(root)


if __name__ == "__main__":
    main()
