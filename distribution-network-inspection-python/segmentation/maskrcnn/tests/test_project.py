import ast
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wireseg.coco_metrics import evaluate_predictions
from wireseg.common import PROJECT_ROOT, read_config, split_paths, write_json
from wireseg.dataset import WireDataset
from wireseg.evaluation import evaluation_identity, reuse_predictions


def coco_fixture():
    gt = COCO()
    gt.dataset = {
        "info": {},
        "images": [{"id": i, "height": 8, "width": 8} for i in range(3)],
        "categories": [{"id": 0, "name": "cable"}],
        "annotations": [],
    }
    mask = np.zeros((8, 8), np.uint8)
    mask[3, 1:7] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    for image_id in (0, 1):
        gt.dataset["annotations"].append(
            {
                "id": image_id + 1,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [1, 3, 6, 1],
                "segmentation": copy.deepcopy(rle),
                "area": 6,
                "iscrowd": 0,
            }
        )
    gt.createIndex()
    prediction = {
        "bbox": [{"image_id": 0, "category_id": 0, "bbox": [1, 3, 6, 1], "score": 0.8}],
        "segm": [
            {
                "image_id": 0,
                "category_id": 0,
                "segmentation": copy.deepcopy(rle),
                "score": 0.8,
            }
        ],
    }
    return gt, prediction


class MetricsRegression(unittest.TestCase):
    def setUp(self):
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)

    def test_thin_gt_empty_predictions_and_negative_frames(self):
        gt, prediction = coco_fixture()
        perfect = evaluate_predictions(gt, prediction, [0])
        missed = evaluate_predictions(gt, prediction, [0, 1])
        false_positive = copy.deepcopy(prediction)
        for rows in false_positive.values():
            rows.append(dict(rows[0], image_id=2, score=0.99))
        negative = evaluate_predictions(gt, false_positive, [0, 1, 2])
        empty = evaluate_predictions(gt, {"bbox": [], "segm": []}, [0, 1, 2])
        for kind in ("bbox", "segm"):
            self.assertAlmostEqual(perfect[kind]["AP"], 100.0)
            self.assertLess(missed[kind]["AP"], 60.0)
            self.assertLess(negative[kind]["AP"], missed[kind]["AP"])
            self.assertEqual(empty[kind]["AP"], 0.0)
            self.assertAlmostEqual(
                missed[kind]["AP"], np.mean(missed[kind]["AP_by_iou"])
            )
            self.assertAlmostEqual(
                missed[kind]["AP"], missed[kind]["per_class"][0]["AP"]
            )

    def test_metric_kernel_is_identical_to_existing_models(self):
        root = PROJECT_ROOT.parent

        def kernel(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "evaluate_predictions"
            )
            return ast.dump(function, include_attributes=False)

        current = kernel(PROJECT_ROOT / "wireseg/coco_metrics.py")
        for project in ("rfdetr", "yolact"):
            self.assertEqual(current, kernel(root / project / "coco_evaluation.py"))


class RealDatasetRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_config()
        paths = split_paths(cls.config, "train")
        if not all(Path(path).exists() for path in paths):
            raise unittest.SkipTest("Requires external full wire dataset")
        cls.dataset = WireDataset(*paths)

    def test_population_and_label_mapping(self):
        expected = {
            "train": (961, 7637, 6),
            "val": (125, 954, 2),
            "test": (229, 2033, 3),
        }
        for split, counts in expected.items():
            dataset = WireDataset(
                *split_paths(self.config, split), include_targets=False
            )
            identity = evaluation_identity(dataset, self.config)
            self.assertEqual(
                (
                    len(dataset),
                    identity["gt_instance_count"],
                    identity["empty_gt_images"],
                ),
                counts,
            )
            self.assertEqual(dataset.label_to_category, {1: 0})

    def test_degenerate_boxes_repaired_without_dropping_instances(self):
        for annotation_id in (5873, 7149):
            image_id = self.dataset.coco.anns[annotation_id]["image_id"]
            _, target = self.dataset[self.dataset.image_ids.index(image_id)]
            self.assertEqual(
                len(target["labels"]),
                len(self.dataset.coco.getAnnIds(imgIds=[image_id])),
            )
            self.assertIn(annotation_id, target["repaired_annotation_ids"].tolist())
            self.assertTrue((target["boxes"][:, 2:] > target["boxes"][:, :2]).all())
            self.assertTrue((target["labels"] == 1).all())

    def test_empty_images_and_subpixel_masks_are_retained(self):
        negatives = [
            i
            for i in self.dataset.image_ids
            if not self.dataset.coco.getAnnIds(imgIds=[i])
        ]
        self.assertEqual(len(negatives), 6)
        for image_id in negatives:
            _, target = self.dataset[self.dataset.image_ids.index(image_id)]
            self.assertEqual(tuple(target["boxes"].shape), (0, 4))
            self.assertEqual(tuple(target["masks"].shape), (0, 700, 700))
        image_id = self.dataset.coco.anns[6178]["image_id"]
        _, target = self.dataset[self.dataset.image_ids.index(image_id)]
        self.assertIn(6178, target["empty_mask_annotation_ids"].tolist())
        self.assertEqual(
            len(target["labels"]), len(self.dataset.coco.getAnnIds(imgIds=[image_id]))
        )

    def test_flip_applies_to_boxes_and_masks_together(self):
        flipped = WireDataset(
            *split_paths(self.config, "train"),
            training=True,
            horizontal_flip_probability=1.0,
        )
        image, target = self.dataset[0]
        mirror, transformed = flipped[0]
        np.testing.assert_array_equal(mirror.numpy(), image.flip(-1).numpy())
        np.testing.assert_array_equal(
            transformed["masks"].numpy(), target["masks"].flip(-1).numpy()
        )
        np.testing.assert_allclose(
            transformed["boxes"][:, 0].numpy(), 700 - target["boxes"][:, 2].numpy()
        )

    def test_reuse_rejects_changed_threshold_before_scoring(self):
        dataset = WireDataset(
            *split_paths(self.config, "test"), include_targets=False, max_images=3
        )
        identity = evaluation_identity(dataset, self.config)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "results") as directory:
            identity["score_threshold"] = 0.2
            write_json(Path(directory) / "maskrcnn_metadata.json", identity)
            with self.assertRaisesRegex(ValueError, "score_threshold"):
                reuse_predictions(dataset, self.config, directory, {})


if __name__ == "__main__":
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    unittest.main(verbosity=2)
