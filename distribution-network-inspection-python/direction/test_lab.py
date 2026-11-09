"""Unit tests for regional direction estimation and annotation handling."""

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np
from wirelab import legacy, probabilistic
from wirelab.io import ROOT, axial_delta, references, read, manifest
from wirelab.evaluation import compare, summarize
from check_kotlin_parity import line_mask


class LabTests(unittest.TestCase):
    def test_known_geometry(self):
        for angle in (-45, -20, 0, 20, 45):
            result = legacy.analyze([(line_mask(angle), 1.0)], 128, 128)
            for region in ("upper", "lower"):
                self.assertTrue(result[region]["options"])
                self.assertLess(
                    abs(
                        float(axial_delta(result[region]["options"][0]["angle"], angle))
                    ),
                    3,
                )

    def test_extension_does_not_rotate_fixed_endpoints(self):
        y, x = np.nonzero(line_mask(30))
        points = np.column_stack([x, y]).astype(float)
        raw = [
            legacy.fit(points, replace(legacy.Config(), extend=e))["raw"]
            for e in (1.0, 1.1, 1.2, 1.4)
        ]
        self.assertLess(np.ptp(raw), 1e-5)

    def test_empty_masks(self):
        result = legacy.analyze([], 128, 128)
        self.assertEqual(result["upper"]["options"], [])
        self.assertEqual(
            probabilistic.analyze([], 128, 128, bootstrap=40)["lower"]["options"], []
        )

    def test_probability_preserves_distinct_modes(self):
        cs = [{"angle": a, "weight": 1.0} for a in (-31, -30, -29, 29, 30, 31)]
        model = probabilistic.selected_model(cs)
        self.assertEqual(len(model["means"]), 2)
        self.assertGreater(abs(float(axial_delta(*model["means"]))), 50)

    def test_bootstrap_reproducible(self):
        masks = [(line_mask(30, offset=i * 7), 0.8) for i in (-1, 0, 1)]
        a = probabilistic.analyze(masks, 128, 128, bootstrap=40, seed=4)
        b = probabilistic.analyze(masks, 128, 128, bootstrap=40, seed=4)
        self.assertEqual(a, b)
        self.assertIsNotNone(a["upper"]["options"][0]["interval"])

    def test_false_output_and_misses_are_counted(self):
        item = {"key": "synthetic", "role": "dev", "source": "synthetic"}
        label = {
            "regions": {"upper": "labeled", "lower": "no_wire"},
            "lines": [
                {
                    "region": "upper",
                    "group": 1,
                    "points": [[64, 60], [64, 10]],
                    "directed": True,
                }
            ],
        }
        result = {
            "upper": {"options": [], "candidate_count": 0, "rejected": {}},
            "lower": {
                "options": [{"angle": 0, "direction": 180}],
                "candidate_count": 1,
                "rejected": {},
            },
        }
        stats = summarize(compare(item, label, result, "probabilistic", "gt"))
        self.assertEqual(stats["direction_recall_at_5deg"], 0.0)
        self.assertEqual(stats["negative_region_false_output_rate"], 1.0)
        self.assertEqual(stats["CI_coverage_all_references"], 0.0)

    def test_assignment_is_one_to_one(self):
        item = {"key": "synthetic", "role": "dev", "source": "synthetic"}
        label = {
            "regions": {"upper": "labeled", "lower": "ambiguous"},
            "lines": [
                {
                    "region": "upper",
                    "group": 1,
                    "points": [[64, 60], [64, 10]],
                    "directed": False,
                }
            ],
        }
        result = {
            "upper": {
                "options": [{"angle": 0, "direction": 0}, {"angle": 1, "direction": 1}],
                "candidate_count": 2,
                "rejected": {},
            }
        }
        stats = summarize(compare(item, label, result, "synthetic", "gt"))
        self.assertEqual(stats["direction_precision_at_5deg"], 0.5)
        self.assertEqual(stats["direction_recall_at_5deg"], 1.0)

    def test_angular_wrap(self):
        self.assertAlmostEqual(abs(float(axial_delta(89, -89))), 2.0)

    def test_group_reference_uses_only_human_lines(self):
        label = {
            "lines": [
                {
                    "region": "upper",
                    "group": 2,
                    "points": [[80, 80], [20, 20]],
                    "directed": True,
                }
            ]
        }
        self.assertAlmostEqual(references(label, "upper")[0]["angle"], 45.0)

    def test_camera_pose_screen_uses_pitch_and_roll_not_wire_orientation(self):
        from prepare_local_dataset import nadir_status

        self.assertEqual(
            nadir_status({"GimbalPitchDegree": -89.9, "GimbalRollDegree": 0}),
            "eligible",
        )
        self.assertEqual(
            nadir_status({"GimbalPitchDegree": -29.3, "GimbalRollDegree": 0}),
            "non_nadir_camera_pose",
        )
        self.assertEqual(
            nadir_status({"GimbalPitchDegree": -90, "GimbalRollDegree": 20}),
            "non_nadir_camera_pose",
        )
        self.assertEqual(nadir_status({}), "missing_camera_metadata")

    def test_external_missing_gt_is_not_treated_as_negative_mask(self):
        from wirelab.io import MaskStore

        store = object.__new__(MaskStore)
        with self.assertRaises(ValueError):
            store.load({"key": "external_fixture", "has_gt_mask": False}, "gt")

    @unittest.skipUnless(
        (ROOT / "dataset/manifest.json").is_file(),
        "Optional GUI check needs external annotation dataset",
    )
    def test_viewpoint_rejection_is_not_no_wire(self):
        import tkinter as tk
        from annotate_angles import App
        from run_experiments import load_labels
        from wirelab.io import write

        base = ROOT / "validation"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(base)) as folder:
            folder = Path(folder)
            window = tk.Tk()
            window.withdraw()
            try:
                app = App(window, ROOT, labels=folder / "annotations")
                app.viewpoint.set("斜视，不适用")
                self.assertTrue(app.save(True))
                rejected = read(app.path(app.item))
                self.assertEqual(rejected["viewpoint"], "oblique")
                self.assertEqual(rejected["regions"]["upper"], "unreviewed")
                first = dict(app.item, role="dev")
                second = dict(
                    first, key="synthetic_valid_frame", image_sha256="fixture"
                )
                write(folder / "dataset/manifest.json", {"images": [first, second]})
                write(
                    folder / "annotations/synthetic_valid_frame.json",
                    {
                        "reviewed": True,
                        "image_sha256": "fixture",
                        "viewpoint": "nadir",
                        "regions": {"upper": "no_wire", "lower": "no_wire"},
                        "lines": [],
                    },
                )
                labeled, missing = load_labels(folder, "dev")
                self.assertEqual(
                    [item["key"] for item, _ in labeled], ["synthetic_valid_frame"]
                )
                self.assertFalse(missing)
            finally:
                window.destroy()

    def test_unlabeled_data_cannot_produce_results(self):
        from run_experiments import load_labels

        base = ROOT / "validation"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(base)) as folder:
            folder = Path(folder)
            self.assertEqual(folder.resolve().parent, base.resolve())
            from wirelab.io import write

            write(
                folder / "dataset/manifest.json",
                {"images": [{"role": "dev", "key": "unlabeled"}]},
            )
            with self.assertRaises(RuntimeError):
                load_labels(folder, "dev")

    @unittest.skipUnless(
        (ROOT / "dataset/manifest.json").is_file(),
        "Optional GUI check needs external annotation dataset",
    )
    def test_gui_roundtrip_uses_temporary_labels_only(self):
        import tkinter as tk
        from annotate_angles import App

        base = ROOT / "validation"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(base)) as folder:
            folder = Path(folder)
            self.assertEqual(folder.resolve().parent, base.resolve())
            window = tk.Tk()
            window.withdraw()
            try:
                app = App(window, ROOT, labels=folder)
                app.points = [[350, 300], [300, 100]]
                app.finish_line()
                self.assertEqual(len(app.doc["lines"]), 1)
                app.state_vars["lower"].set("无可见导线")
                self.assertTrue(app.save(True))
                saved = read(app.path(app.item))
                self.assertTrue(saved["reviewed"])
                self.assertEqual(saved["regions"]["upper"], "labeled")
                self.assertEqual(saved["lines"][0]["points"], [[350, 300], [300, 100]])
            finally:
                window.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
