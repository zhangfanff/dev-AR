# -*- coding: utf-8 -*-
"""Copy manually reviewed, local-only TEST subsets; never change source data."""

import csv, hashlib, html, json, math, shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

WORK = Path(__file__).resolve().parents[2] / "."
REVIEW = WORK / "work/scene_review_20261102"
OUT = Path("G:/pythonProjects/reviewer2_daytime_scene_subsets_20261102")
LAB = Path(__file__).resolve().parents[2] / "direction"
inventory = json.loads((REVIEW / "inventory.json").read_text(encoding="utf-8"))
bykey = {r["key"]: r for r in inventory}
SCENES = {
    "01_built_up_background": {
        "name": "建筑背景复杂",
        "definition": "目标周围有多处房屋、屋顶、厂房或沿街建筑，建筑边缘及附属设施构成背景干扰；密度在 village_dense / industrial / street_built_up 字段中另行说明。",
        "members": {
            "tower": [
                "tower_037",
                "tower_041",
                "tower_044",
                "tower_048",
                "tower_308",
                "tower_321",
                "tower_333",
                "tower_334",
                "tower_348",
                "tower_349",
                "tower_367",
                "tower_416",
            ],
            "tower_top": [
                "tower_top_100",
                "tower_top_105",
                "tower_top_121",
                "tower_top_123",
                "tower_top_127",
                "tower_top_128",
                "tower_top_140",
            ],
        },
    },
    "02_strong_reflection": {
        "name": "强反光／高亮背景",
        "definition": "画面存在可见的水面或塑料棚膜镜面反射、高亮背景与较暗目标的明暗反差。依据可观察图像特征分组，不根据像素均值推断照度或太阳方位。",
        "members": {
            "tower": ["tower_062", "tower_163", "tower_344"],
            "tower_top": ["tower_top_036", "tower_top_037"],
        },
    },
    "03_vegetation_interference": {
        "name": "植被干扰（区分局部遮挡与植被背景）",
        "definition": "目标周围存在密集枝叶、竹丛或树冠，与目标或细导线相邻、投影重叠，或使局部目标难以辨认。遮挡证据与单纯背景干扰分别记录，不把整个组写成严格的物理遮挡组。",
        "members": {
            "tower": ["tower_007", "tower_014", "tower_021", "tower_074", "tower_299"],
            "tower_top": [
                "tower_top_007",
                "tower_top_033",
                "tower_top_045",
                "tower_top_055",
                "tower_top_058",
                "tower_top_064",
            ],
            "wire": [f"wire_{i}" for i in range(220, 229)],
            "angle": [],
        },
    },
}
for r in inventory:
    if r["task"] == "angle":
        a = json.loads(Path(r["annotation_path"]).read_text(encoding="utf-8"))
        if a.get("reviewed") and "植被遮挡" in a.get("tags", []):
            SCENES["03_vegetation_interference"]["members"]["angle"].append(r["key"])

NOTES = {
    "tower_037": "沿线路连续村镇建筑，屋顶和墙体边缘较多。",
    "tower_041": "道路两侧成片村镇住宅，目标与屋顶背景重叠。",
    "tower_044": "连续住宅与道路构成多层次背景。",
    "tower_048": "多栋厂房、住宅临近杆塔，屋顶及墙体边缘密集。",
    "tower_308": "目标位于多栋建筑之间的道路。",
    "tower_321": "近处房屋与目标框投影相邻，多栋建筑填充背景。",
    "tower_333": "连续村镇房屋、屋顶和沿街设施形成复杂背景。",
    "tower_334": "目标与多栋房屋、墙面和多组线路投影重叠。",
    "tower_348": "沿街建筑及屋顶与杆塔、导线叠加。",
    "tower_349": "目标紧邻连续房屋，墙体与线缆背景复杂。",
    "tower_367": "村镇路段，近处屋顶及连续房屋形成背景。",
    "tower_416": "道路旁多处建筑、围墙、车辆及线路叠加。",
    "tower_top_100": "厂房、围挡和设施构成工业建筑背景；不等同高层密集城区。",
    "tower_top_105": "多处低层房屋、屋顶、墙面与多组导线叠加。",
    "tower_top_121": "连续低层建筑及附属设施形成背景。",
    "tower_top_123": "厂房、院落和堆料形成建筑设施背景。",
    "tower_top_127": "多栋相邻房屋、屋顶及围墙形成背景。",
    "tower_top_128": "厂房、机械与导线构成工业建筑背景。",
    "tower_top_140": "沿街建筑、停放车辆及树冠与目标邻近。",
    "tower_062": "塑料棚膜出现明显高亮反射，前景杆塔相对偏暗。",
    "tower_163": "目标后方水面存在明显镜面光斑。",
    "tower_344": "水面强反光与较暗的杆塔、路面形成较高明暗反差。",
    "tower_top_036": "目标周围塑料棚膜有高亮反射条带。",
    "tower_top_037": "画面右下棚膜有明显高亮反射区。",
    "tower_007": "杆体下段与竹丛重叠，局部下段可见性受影响。",
    "tower_014": "杆体下段位于枝叶前后叠加区域，末端局部被植被遮挡。",
    "tower_021": "杆体下段同时受枝叶和屋顶遮挡，属于混合遮挡。",
    "tower_074": "目标下段与竹丛、枝叶叠加；局部遮挡主要在杆体下段。",
    "tower_299": "杆体下部与高密度树冠重叠，局部结构被枝叶遮挡。",
    "tower_top_007": "塔顶目标周围为密集植被，主要为植被背景干扰。",
    "tower_top_033": "多根杆塔位于密集树冠背景上，塔顶本身仍可见。",
    "tower_top_045": "塔顶周围有复杂枝叶纹理，主要为背景干扰。",
    "tower_top_055": "小尺度塔顶位于密集枝叶背景前，主要为尺度与植被背景干扰。",
    "tower_top_058": "近处塔顶与高亮树冠投影相邻，主要为植被背景干扰。",
    "tower_top_064": "细杆体及小塔顶位于茂密枝叶背景上，主要为背景干扰。",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def cp(source, dest):
    source = Path(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    before = sha(source)
    if dest.exists():
        assert sha(dest) == before, f"Existing output differs: {dest}"
    else:
        shutil.copy2(source, dest)
    assert sha(dest) == before and sha(source) == before
    return before


def labels(path):
    data = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        z = list(map(float, line.split()))
        assert len(z) == 5
        assert (
            z[0].is_integer()
            and all(0 <= v <= 1 for v in z[1:])
            and z[3] > 0
            and z[4] > 0
        )
        data.append(z)
    return data


original_coco = {
    task: json.loads(
        (
            WORK
            / "outputs/detector_evaluation_20261020"
            / f"{task}_yolov12n/ground_truth_coco.json"
        ).read_text(encoding="utf-8")
    )
    for task in ["tower", "tower_top"]
}
original_coco["wire"] = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "segmentation/rfdetr/dataset_new/test/_annotations.coco.json"
    ).read_text(encoding="utf-8")
)
index = []
summaries = []
fingerprints = {}
OUT.mkdir(parents=True, exist_ok=True)
for scene, setting in SCENES.items():
    for task, keys in setting["members"].items():
        if not keys:
            continue
        group = OUT / scene / task
        images = group / "images"
        group_rows = []
        ids = set()
        newnames = {}
        angle_manifest = []
        for key in keys:
            r = bykey[key]
            src = Path(r["image_path"])
            assert r["source_split"] == "test" or (
                task == "angle"
                and r["manifest_item"]["role"] == "test"
                and not r["manifest_item"]["segmentation_training_seen"]
            )
            if task == "angle":
                name = r["manifest_item"]["key"] + ".jpg"
            else:
                name = src.name
            dest = images / name
            digest = cp(src, dest)
            fingerprints[str(src)] = digest
            row = dict(
                scene=scene,
                scene_name=setting["name"],
                task=task,
                key=key,
                source_image=str(src),
                source_split=r["source_split"],
                evaluation_role="test",
                copied_image=dest.relative_to(OUT).as_posix(),
                image_sha256=digest,
                source_local=True,
                selection_uses_model_outputs=False,
                scene_basis=NOTES.get(key, ""),
                occlusion_evidence="not_assessed",
                mask_ground_truth=False,
                direction_ground_truth=False,
            )
            if scene == "01_built_up_background":
                row["building_type"] = (
                    "industrial"
                    if key in ["tower_top_100", "tower_top_123", "tower_top_128"]
                    else "street_built_up"
                    if key == "tower_top_140"
                    else "village_dense"
                )
            if task in ["tower", "tower_top"]:
                source_label = Path(r["label_path"])
                destination_label = group / "labels" / Path(name).with_suffix(".txt")
                ldigest = cp(source_label, destination_label)
                fingerprints[str(source_label)] = ldigest
                rows = labels(source_label)
                assert any(v[0] == 0 for v in rows)
                row.update(
                    source_annotation=str(source_label),
                    copied_annotation=destination_label.relative_to(OUT).as_posix(),
                    annotation_sha256=ldigest,
                    instance_count=len(rows),
                    target_instance_count=sum(v[0] == 0 for v in rows),
                )
                gt = original_coco[task]
                im = next(
                    im for im in gt["images"] if Path(im["file_name"]).name == src.name
                )
                # Check the existing evaluation ground truth still corresponds to the live labels.
                w, h = Image.open(src).size
                assert (w, h) == (im["width"], im["height"])
                expected = Counter(
                    (int(v[0]),)
                    + tuple(
                        round(z, 3)
                        for z in [
                            (v[1] - v[3] / 2) * w,
                            (v[2] - v[4] / 2) * h,
                            v[3] * w,
                            v[4] * h,
                        ]
                    )
                    for v in rows
                )
                cat_to_class = {c["id"]: i for i, c in enumerate(gt["categories"])}
                actual = Counter(
                    (cat_to_class[a["category_id"]],)
                    + tuple(round(z, 3) for z in a["bbox"])
                    for a in gt["annotations"]
                    if a["image_id"] == im["id"]
                )
                assert actual == expected, (
                    f"Cached COCO differs from live labels: {key}"
                )
                row["original_image_id"] = im["id"]
                ids.add(im["id"])
                newnames[im["id"]] = "images/" + name
                if scene == "03_vegetation_interference":
                    row["occlusion_evidence"] = (
                        "partial_lower_body_occlusion_mixed_with_building"
                        if key == "tower_021"
                        else "partial_lower_body_occlusion"
                        if task == "tower"
                        else "vegetation_background_only"
                    )
            elif task == "wire":
                assert src.name.upper().startswith("DJI")
                ids.add(r["image_id"])
                newnames[r["image_id"]] = "images/" + name
                anns = [
                    a
                    for a in original_coco["wire"]["annotations"]
                    if a["image_id"] == r["image_id"]
                ]
                row.update(
                    original_image_id=r["image_id"],
                    mask_ground_truth=True,
                    instance_count=len(anns),
                    target_instance_count=len(anns),
                    source_annotation="G:/pythonProjects/RF-DETR/dataset_new/test/_annotations.coco.json",
                    copied_annotation=(group / "annotations_coco.json")
                    .relative_to(OUT)
                    .as_posix(),
                )
                row["scene_basis"] = (
                    "自采导线测试图像，包含树冠或密集植被背景。"
                    if r["image_id"] < 225
                    else "自采导线测试图像，树冠、枝叶与细导线投影重叠或相邻，存在局部可见性干扰。"
                )
                row["occlusion_evidence"] = (
                    "vegetation_projection_overlap_visibility_ambiguous"
                    if r["image_id"] in [225, 226, 227]
                    else "vegetation_background_only"
                )
            else:
                item = r["manifest_item"]
                assert (
                    item["role"] == "test"
                    and item["source"] == "local"
                    and not item["segmentation_training_seen"]
                )
                a = json.loads(Path(r["annotation_path"]).read_text(encoding="utf-8"))
                assert a["reviewed"] and a["image_sha256"] == digest
                assert (
                    abs(item["camera_metadata"]["GimbalPitchDegree"] + 90) <= 5
                    and abs(item["camera_metadata"]["GimbalRollDegree"]) <= 5
                )
                copied_annotation = (
                    group / "annotations" / Path(r["annotation_path"]).name
                )
                adigest = cp(r["annotation_path"], copied_annotation)
                fingerprints[r["annotation_path"]] = adigest
                nitem = dict(item)
                nitem["image_path"] = "images/" + name
                angle_manifest.append(nitem)
                row.update(
                    direction_ground_truth=True,
                    mask_ground_truth=item["has_gt_mask"],
                    source_annotation=r["annotation_path"],
                    copied_annotation=copied_annotation.relative_to(OUT).as_posix(),
                    annotation_sha256=adigest,
                    reference_polyline_count=len(a["lines"]),
                    original_manual_scene_tags=a.get("tags", []),
                    scene_basis="沿用人工方向标注中的植被遮挡标签，并复核为俯视植被场景；部分为复杂树冠背景，不单独推断物理遮挡。",
                    occlusion_evidence="manual_vegetation_tag_background_or_projection_overlap",
                    source_camera_metadata=item["camera_metadata"],
                    source_field_image=item["source_path"],
                )
            group_rows.append(row)
            index.append(row)
        if task != "angle":
            gt = original_coco[task]
            subset = {k: v for k, v in gt.items() if k not in ["images", "annotations"]}
            subset["images"] = [
                dict(im, file_name=newnames[im["id"]])
                for im in gt["images"]
                if im["id"] in ids
            ]
            subset["annotations"] = [
                a for a in gt["annotations"] if a["image_id"] in ids
            ]
            dump(group / "annotations_coco.json", subset)
            anncount = len(subset["annotations"])
            for row in group_rows:
                assert (
                    sum(
                        a["image_id"] == row["original_image_id"]
                        for a in subset["annotations"]
                    )
                    == row["instance_count"]
                )
            if task in ["tower", "tower_top"]:
                names = (
                    ["tower"]
                    if task == "tower"
                    else ["tower-top", "hengdan", "jinggai", "car", "byq"]
                )
                config = (
                    f"path: {json.dumps(group.as_posix())}\ntrain: null\nval: images\ntest: images\nnames:\n"
                    + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
                )
                (group / "data.yaml").write_text(config, encoding="utf-8")
        else:
            anncount = sum(r["reference_polyline_count"] for r in group_rows)
            dump(
                group / "manifest.json",
                {
                    "schema": "reviewer2_direction_test_subset_v1",
                    "source_manifest": str(LAB / "dataset/manifest.json"),
                    "role": "test",
                    "source": "local",
                    "images": angle_manifest,
                    "direction_labels_only": all(
                        not x["has_gt_mask"] for x in angle_manifest
                    ),
                },
            )
        dump(group / "source_index.json", group_rows)
        (group / "test_images.txt").write_text(
            "".join((OUT / r["copied_image"]).as_posix() + "\n" for r in group_rows),
            encoding="utf-8",
        )
        summaries.append(
            dict(
                scene=scene,
                scene_name=setting["name"],
                task=task,
                images=len(group_rows),
                annotation_count=anncount,
                annotation_unit="direction_polylines"
                if task == "angle"
                else "instances",
                target_instances=sum(
                    r.get("target_instance_count", 0) for r in group_rows
                )
                if task != "angle"
                else None,
            )
        )

dump(OUT / "scene_definitions.json", SCENES)
dump(OUT / "source_index.json", index)
dump(
    OUT / "summary.json",
    {
        "created_at": datetime.now().isoformat(),
        "output_root": str(OUT),
        "inventory": Counter(r["task"] for r in inventory),
        "groups": summaries,
        "selected_records": len(index),
        "unique_image_hashes": len({r["image_sha256"] for r in index}),
        "source_change_policy": "copy_only",
        "ttpla_images": 0,
        "selection_uses_prediction_quality": False,
        "strict_backlighting_group_created": False,
        "daytime_vegetation_occlusion_is_not_equal_to_all_vegetation_backgrounds": True,
    },
)
fields = [
    "scene_name",
    "task",
    "key",
    "source_split",
    "evaluation_role",
    "source_image",
    "source_annotation",
    "copied_image",
    "copied_annotation",
    "image_sha256",
    "occlusion_evidence",
    "scene_basis",
]
with (OUT / "样本来源与分组.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(index)

# Contact sheets are browsing previews only. All evaluation images remain byte-identical.
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
(OUT / "previews").mkdir(exist_ok=True)
gallery = [
    '<!doctype html><html lang="zh"><meta charset="utf-8"><title>日间困难场景测试子集</title><style>body{font:16px sans-serif;margin:24px;background:#f5f5f5}section{display:flex;flex-wrap:wrap;gap:14px}article{width:300px;background:white;padding:10px}img{width:300px;height:230px;object-fit:contain}small{overflow-wrap:anywhere}</style><h1>日间困难场景测试子集</h1><p>图片均来自现有自采测试数据。点击缩略图打开原图；本页面仅用于浏览，不是模型结果展示。</p>'
]
for scene, setting in SCENES.items():
    rows = [r for r in index if r["scene"] == scene]
    gallery.append(
        f"<h2>{html.escape(setting['name'])}（{len(rows)} 条任务样本）</h2><p>{html.escape(setting['definition'])}</p><section>"
    )
    for page in range(math.ceil(len(rows) / 12)):
        sub = rows[page * 12 : (page + 1) * 12]
        sheet = Image.new("RGB", (4 * 340, math.ceil(len(sub) / 4) * 300), "white")
        d = ImageDraw.Draw(sheet)
        for j, r in enumerate(sub):
            im = ImageOps.contain(
                Image.open(OUT / r["copied_image"]).convert("RGB"), (334, 249)
            )
            x = j % 4 * 340
            y = j // 4 * 300
            sheet.paste(im, (x, y))
            d.text(
                (x + 3, y + 252),
                r["key"].replace("angle_local_", "角度_"),
                font=font,
                fill="black",
            )
            d.text((x + 3, y + 274), r["task"], font=font, fill="black")
        sheet.save(OUT / "previews" / f"{scene}_{page + 1}.jpg", quality=93)
    for r in rows:
        path = html.escape(r["copied_image"], quote=True)
        gallery.append(
            f'<article><a href="{path}"><img loading="lazy" src="{path}"></a><b>{html.escape(r["key"])}</b><p>{html.escape(r["scene_basis"])}</p><small>{html.escape(r["occlusion_evidence"])}</small></article>'
        )
    gallery.append("</section>")
(OUT / "打开查看分组.html").write_text("\n".join(gallery) + "</html>", encoding="utf-8")

table = "\n".join(
    f"|{s['scene_name']}|{s['task']}|{s['images']}|{s['annotation_count']} {s['annotation_unit']}|"
    for s in summaries
)
readme = f"""# 日间场景子集

共 {len(index)} 条分组记录，按文件 SHA256 去重后 {len({r["image_sha256"] for r in index})} 张图像。

- images：评测图像。
- previews：浏览缩略图。
- 样本来源与分组.csv：来源、场景标签及分组依据。
- 检测与分割使用原标注；方向任务使用人工方向标注。

各组允许重叠，整体数量按图像去重。植被背景干扰与物理遮挡通过场景标签区分。分组样本保持原测试身份，不参与训练和参数选择。
"""
(OUT / "先看这里.md").write_text(readme, encoding="utf-8")

# Preserve extraction recipe and the exact, immutable selection.
shutil.copy2(__file__, OUT / "prepare_scenes_initial.py")
dump(OUT / "source_fingerprints.json", fingerprints)
assert all(sha(path) == digest for path, digest in fingerprints.items())
dump(
    OUT / "verification.json",
    {
        "source_files_verified_unchanged": len(fingerprints),
        "copied_images_verified": len(index),
        "all_images_from_test_roles": True,
        "all_wire_images_local_DJI": True,
        "direction_images_test_nadir_and_unseen": True,
        "detection_live_labels_match_existing_test_COCO": True,
        "all_copied_labels_and_images_byte_identical": True,
        "paper_modified": False,
        "metrics_computed": False,
    },
)
print(
    json.dumps(
        {
            "root": str(OUT),
            "groups": summaries,
            "records": len(index),
            "unique_images": len({r["image_sha256"] for r in index}),
        },
        ensure_ascii=True,
        indent=2,
    )
)
