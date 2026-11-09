import unittest, tempfile, json, csv
from pathlib import Path
import numpy as np
from PIL import Image
from evaluate_resolutions import preprocess, nms, decode_yolo, dataset
from summarize_runs import percentile, summarize


class Tests(unittest.TestCase):
    def test_letterbox_and_inverse(self):
        cfg = {"width": 640, "task": "tower", "iou": 0.7}
        x, g = preprocess(Image.new("RGB", (1280, 720), (255, 0, 0)), cfg)
        self.assertEqual(x.shape, (1, 3, 640, 640))
        self.assertEqual(g, (0.5, 0, 140))
        raw = np.array([[[320], [320], [320], [180], [0.9]]], dtype=np.float32)
        b, s, c = decode_yolo(raw, cfg, g, 1280, 720)
        np.testing.assert_allclose(b, [[320, 180, 960, 540]])

    def test_class_aware_nms(self):
        boxes = np.array([[0, 0, 10, 10]] * 3, dtype=float)
        self.assertEqual(
            nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1])), [0, 2]
        )

    def test_wire_normalization(self):
        x, _ = preprocess(
            Image.new("RGB", (20, 10), (255, 0, 0)), {"width": 24, "task": "wire"}
        )
        np.testing.assert_allclose(
            x[0, :, 0, 0],
            [(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            rtol=1e-6,
        )

    def test_percentile(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.95), 3.85)
        self.assertIsNone(percentile([], 0.5))

    @unittest.skipUnless(
        __import__("os").environ.get("RUN_EXTERNAL_DATA_TESTS") == "1",
        "Requires external full test datasets",
    )
    def test_test_datasets(self):
        for task, count in [("tower", 462), ("tower_top", 142), ("wire", 229)]:
            gt, paths = dataset(task)
            self.assertEqual(len(paths), count)
            self.assertTrue(all(p.is_file() for p in paths.values()))

    def test_missing_gpu_and_charging_power(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp)
            (f / "summary.json").write_text(
                json.dumps(
                    dict(
                        model="test",
                        received=10,
                        processed=8,
                        sampled_out=0,
                        overload_dropped=2,
                        errors=0,
                        warmup_frames=2,
                        completed_fps=8,
                        baseline=False,
                    )
                )
            )
            (f / "frames.csv").write_text(
                "preprocess_ms,inference_ms,postprocess_ms,callback_to_result_ms\n1,2,3,6\n"
            )
            (f / "telemetry.jsonl").write_text(
                json.dumps(
                    dict(
                        phase="measurement",
                        gpu_busy_percent=None,
                        battery_power_magnitude_W=2,
                        battery_plugged=1,
                        battery_status=2,
                    )
                )
            )
            s = summarize(f, 5, 20)
            self.assertIsNone(s["gpu_busy_percent_mean"])
            self.assertIsNone(s["discharging_battery_power_W_mean"])
            self.assertTrue(s["meets_realtime_requirement"])


if __name__ == "__main__":
    unittest.main()
