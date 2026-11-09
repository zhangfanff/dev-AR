"""Summarize exported controller runs without replacing unsupported sensors with zero."""

from pathlib import Path
import argparse, csv, json, statistics


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    q = (len(xs) - 1) * p
    i = int(q)
    j = min(i + 1, len(xs) - 1)
    return xs[i] + (xs[j] - xs[i]) * (q - i)


def mean(xs):
    return statistics.mean(xs) if xs else None


def summarize(folder, target_fps=None, deadline_ms=None):
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with (folder / "frames.csv").open(encoding="utf-8") as f:
        frames = list(csv.DictReader(f))
    all_telemetry = [
        json.loads(l)
        for l in (folder / "telemetry.jsonl").read_text().splitlines()
        if l.strip()
    ]
    tele = [r for r in all_telemetry if r.get("phase") == "measurement"]

    def values(key, source=tele):
        return [r[key] for r in source if isinstance(r.get(key), (float, int))]

    row = {
        k: s.get(k)
        for k in [
            "model",
            "provider_requested",
            "device",
            "android",
            "threads",
            "duration_s",
            "actual_measurement_s",
            "target_fps",
            "received",
            "processed",
            "sampled_out",
            "overload_dropped",
            "errors",
            "source_callback_fps",
            "completed_fps",
        ]
    }
    config = s.get("model_config", {})
    row["resolution"] = config.get("width")
    row["onnx_sha256"] = config.get("onnx_sha256")
    for col in [
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "callback_to_result_ms",
    ]:
        xs = [float(r[col]) for r in frames]
        row[col + "_mean"] = mean(xs)
        row[col + "_p50"] = percentile(xs, 0.5)
        row[col + "_p95"] = percentile(xs, 0.95)
        row[col + "_p99"] = percentile(xs, 0.99)
    row["overload_drop_percent"] = (
        100 * s["overload_dropped"] / max(1, s["received"] - s["sampled_out"])
    )
    for key in [
        "cpu_one_core_percent",
        "cpu_capacity_percent",
        "gpu_busy_percent",
        "pss_kb",
        "battery_power_magnitude_W",
        "battery_temperature_C",
    ]:
        row[key + "_mean"] = mean(values(key))
        row[key + "_max"] = max(values(key), default=None)
    temperatures = values("battery_temperature_C", all_telemetry)
    row["battery_temperature_rise_final_C"] = (
        temperatures[-1] - temperatures[0] if len(temperatures) > 1 else None
    )
    row["battery_temperature_rise_peak_C"] = (
        max(temperatures) - temperatures[0] if temperatures else None
    )
    # Battery power is only defensible while discharging; plugged/charging readings retained separately.
    discharging = [
        r
        for r in tele
        if r.get("battery_plugged") == 0 and r.get("battery_status") == 3
    ]
    row["discharging_battery_power_W_mean"] = mean(
        values("battery_power_magnitude_W", discharging)
    )
    row["gpu_telemetry_available"] = bool(values("gpu_busy_percent"))
    row["valid_run"] = bool(
        s.get("received", 0) > 0
        and (
            s.get("baseline")
            or (len(frames) > 0 and s.get("warmup_frames", 0) > 0 and s["errors"] == 0)
        )
    )
    if target_fps and deadline_ms:
        row["required_fps"] = target_fps
        row["latency_deadline_ms"] = deadline_ms
        row["meets_realtime_requirement"] = bool(
            row["valid_run"]
            and not s.get("baseline")
            and s.get("completed_fps", 0) >= target_fps
            and row["callback_to_result_ms_p95"] is not None
            and row["callback_to_result_ms_p95"] <= deadline_ms
        )
    row["run_folder"] = str(folder)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path)
    p.add_argument("--out", type=Path, default=Path("controller_summary.csv"))
    p.add_argument("--required-fps", type=float)
    p.add_argument("--deadline-ms", type=float)
    a = p.parse_args()
    if bool(a.required_fps) != bool(a.deadline_ms):
        p.error(
            "Specify both --required-fps and --deadline-ms to define a real-time boundary"
        )
    rows = [
        summarize(f.parent, a.required_fps, a.deadline_ms)
        for f in sorted(a.runs.rglob("summary.json"))
    ]
    if not rows:
        raise FileNotFoundError(
            "No summary.json found; stop/save runs and unzip the exported archive"
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with a.out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    a.out.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(a.out)


if __name__ == "__main__":
    main()
