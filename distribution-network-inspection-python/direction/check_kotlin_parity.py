"""Compare the Python estimator with an external DPCA Kotlin harness."""

from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
from wirelab.io import ROOT, MaskStore, write, sha
from wirelab import legacy


def line_mask(angle, size=128, offset=0):
    y, x = np.mgrid[:size, :size]
    x = x - size / 2 - offset
    y = y - size / 2
    a = np.deg2rad(angle)
    return (np.abs(x * np.cos(a) - y * np.sin(a)) <= 1.5) & (
        np.abs(-x * np.sin(a) - y * np.cos(a)) < size * 0.44
    )


def cases():
    result = []
    for angle in (
        -89,
        -85,
        -76,
        -75,
        -74,
        -45,
        -30,
        -15,
        -0.5,
        0,
        0.5,
        15,
        30,
        45,
        74,
        75,
        76,
        85,
        89,
        90,
    ):
        result.append((f"synthetic_axis_{angle}", [(line_mask(angle), 0.8)], 128, 128))
    for angles in (
        (-30, -30, -30, 30, 30, 30),
        (-89, 89, -88, 88),
        (0, 0, 1, 2),
        (10, 20, 30, 40),
    ):
        result.append(
            (
                "synthetic_groups_" + str(angles),
                [
                    (line_mask(a, offset=(i % 3 - 1) * 8), 0.5 + i * 0.05)
                    for i, a in enumerate(angles)
                ],
                128,
                128,
            )
        )
    result.append(("synthetic_empty", [], 128, 128))
    result.append(("synthetic_blob", [(np.ones((128, 128), bool), 1.0)], 128, 128))
    store = MaskStore()
    items = [x for x in store.m["images"] if x["role"] == "test"]
    # Local test images only. This is a numerical port check, not an accuracy evaluation.
    selected = [x for x in items if x["source"] == "local" and x["key"] in store.pred]
    for item in selected:
        result.append(
            (item["key"], store.load(item, "pred"), item["width"], item["height"])
        )
    return result


def encode(testcases):
    parts = [str(len(testcases))]
    for _, masks, w, h in testcases:
        parts.append(f"{w} {h} {len(masks)}")
        for mask, score in masks:
            padded = np.r_[False, mask.ravel(), False].astype(np.int8)
            edges = np.flatnonzero(np.diff(padded))
            starts = edges[::2]
            lengths = edges[1::2] - starts
            pairs = " ".join(f"{s} {n}" for s, n in zip(starts, lengths))
            parts.append(f"{score} 0 {len(starts)} {pairs}")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--java", default="java")
    p.add_argument(
        "--harness",
        type=Path,
        required=True,
        help="External compiled DPCA Kotlin harness",
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="External WireOffsetCalculatorDPCA.kt reference",
    )
    p.add_argument(
        "--stdlib", required=True, help="External Kotlin standard-library JAR"
    )
    args = p.parse_args()
    data = cases()
    jar = args.harness
    result = subprocess.run(
        [
            args.java,
            "-Xmx1024m",
            "-cp",
            str(jar) + __import__("os").pathsep + args.stdlib,
            "HarnessKt",
        ],
        input=encode(data),
        text=True,
        capture_output=True,
        check=True,
    )
    kotlin = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    assert len(kotlin) == len(data)
    checked = []
    for (name, masks, w, h), target in zip(data, kotlin):
        actual = legacy.analyze(masks, w, h)
        maximum = 0.0
        for region in ("upper", "lower"):
            a, b = actual[region], target[region]
            assert a["candidate_count"] == b["candidate_count"], (name, region, a, b)
            assert len(a["options"]) == len(b["options"]), (name, region, a, b)
            for x, y in zip(a["options"], b["options"]):
                assert x["vote_count"] == y["vote_count"], (name, region, x, y)
                for key in ("angle", "direction"):
                    error = abs(x[key] - y[key])
                    maximum = max(maximum, error)
                    assert error < 0.001, (name, region, key, x, y, error)
        checked.append(
            {"case": name, "maximum_numeric_difference_deg": maximum, "passed": True}
        )
    report = {
        "source_sha256": sha(args.source),
        "python_port_sha256": sha(ROOT / "wirelab/legacy.py"),
        "case_count": len(checked),
        "passed": True,
        "tolerance_deg": 0.001,
        "maximum_difference_deg": max(
            x["maximum_numeric_difference_deg"] for x in checked
        ),
        "scope": "Algorithm equivalence on identical masks; OpenCV Point and logging are replaced by inert JVM stubs. Not Android ONNX or flight testing.",
        "cases": checked,
    }
    write(ROOT / "validation/kotlin_python_parity.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
