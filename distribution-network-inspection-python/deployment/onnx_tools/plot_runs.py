"""Create standalone sustained-run latency/resource plots from a controller export."""

from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    a = p.parse_args()
    with (a.run / "frames.csv").open() as f:
        frames = list(csv.DictReader(f))
    ts = [
        json.loads(x)
        for x in (a.run / "telemetry.jsonl").read_text().splitlines()
        if x.strip()
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    if frames:
        origin = float(frames[0]["arrival_elapsed_ns"])
        x = np.array([float(f["arrival_elapsed_ns"]) for f in frames])
        x = (x - origin) / 1e9
        axes[0, 0].plot(x, [float(f["callback_to_result_ms"]) for f in frames], lw=0.5)
        axes[0, 0].set_ylabel("Decoded-frame to result (ms)")
        bins = np.arange(0, max(2, float(x.max()) + 2), 1)
        counts, edges = np.histogram(x, bins=bins)
        axes[0, 1].plot(edges[:-1], counts)
        axes[0, 1].set_ylabel("Processed frames / second")
    if ts:
        x = np.array([r["elapsed_realtime_ms"] for r in ts])
        x = (x - x[0]) / 1000
        for key, label in [
            ("cpu_one_core_percent", "Process CPU (one core = 100%)"),
            ("gpu_busy_percent", "Device GPU busy"),
        ]:
            vals = [
                r.get(key, np.nan) if r.get(key) is not None else np.nan for r in ts
            ]
            if np.isfinite(vals).any():
                axes[1, 0].plot(x, vals, label=label)
        axes[1, 0].set_ylabel("Utilization (%)")
        axes[1, 0].legend()
        vals = [r.get("battery_temperature_C", np.nan) for r in ts]
        axes[1, 1].plot(x, vals)
        axes[1, 1].set_ylabel("Battery temperature (°C)")
    for ax in axes.flat:
        ax.set_xlabel("Elapsed time (s)")
        ax.grid(alpha=0.2)
    fig.savefig(a.run / "sustained_run.png", dpi=200)
    fig.savefig(a.run / "sustained_run.pdf")
    print(a.run / "sustained_run.png")


if __name__ == "__main__":
    main()
