"""ADB install/model upload/download and optional device-side system telemetry."""

from pathlib import Path
import argparse, json, subprocess, time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "com.inspection.onnxbenchmark"
REMOTE = "/sdcard/Android/data/" + PACKAGE + "/files"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["install", "push", "pull", "snapshot"])
    p.add_argument("--adb", default="adb")
    p.add_argument(
        "--apk",
        type=Path,
        help="External prebuilt benchmark APK; Android sources are not bundled",
    )
    p.add_argument("--serial")
    p.add_argument("--out", type=Path, default=ROOT / "device_results")
    a = p.parse_args()
    base = [a.adb] + (["-s", a.serial] if a.serial else [])

    def call(*args, **kw):
        return subprocess.run(base + list(args), check=True, **kw)

    if a.action == "install":
        if a.apk is None or not a.apk.is_file():
            p.error("Provide --apk with an existing external APK")
        call("install", "-r", str(a.apk))
    elif a.action == "push":
        call("shell", "mkdir", "-p", REMOTE + "/models")
        for folder in sorted((ROOT / "models").iterdir()):
            if (folder / "model.json").is_file():
                call("push", str(folder), REMOTE + "/models/")
    elif a.action == "pull":
        a.out.mkdir(parents=True, exist_ok=True)
        call("pull", REMOTE + "/runs", str(a.out))
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        for name, command in [
            ("thermal", "dumpsys thermalservice"),
            ("battery", "dumpsys battery"),
            ("cpu", "dumpsys cpuinfo"),
            ("hardware", "getprop"),
            ("gpu", "cat /sys/class/kgsl/kgsl-3d0/gpubusy"),
        ]:
            r = subprocess.run(
                base + ["shell", command], capture_output=True, text=True
            )
            (a.out / (name + ".txt")).write_text(
                r.stdout + "\nSTDERR:\n" + r.stderr, encoding="utf-8"
            )
        print(
            "Vendor GPU access may be denied. Missing counters require external profiling, not a zero value."
        )


if __name__ == "__main__":
    main()
