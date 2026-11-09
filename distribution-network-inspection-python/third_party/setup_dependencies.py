"""Download fixed framework sources and apply project compatibility patches."""

from pathlib import Path
import argparse
import hashlib
import io
import json
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", choices=["yolov12", "yolov5", "yolact", "rfdetr_compat"]
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Optional directory of downloaded <profile>.zip archives",
    )
    args = parser.parse_args()
    lock = json.loads(
        (ROOT / "third_party/dependencies.json").read_text(encoding="utf-8")
    )
    for name in args.only or lock:
        info = lock[name]
        folder = ROOT / ".external" / name
        if folder.exists():
            raise FileExistsError(
                f"{folder} already exists; choose a fresh checkout or inspect it before updating"
            )
        if args.archive_dir:
            raw = (args.archive_dir / (name + ".zip")).read_bytes()
        else:
            with urllib.request.urlopen(info["archive_url"], timeout=60) as response:
                raw = response.read()
        if hashlib.sha256(raw).hexdigest() != info["archive_sha256"]:
            raise ValueError(f"Archive checksum mismatch: {name}")
        folder.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            prefix = archive.namelist()[0].split("/")[0] + "/"
            for member in archive.infolist():
                relative = member.filename.removeprefix(prefix)
                if member.is_dir() or not relative:
                    continue
                path = (folder / relative).resolve()
                if not path.is_relative_to(folder.resolve()):
                    raise ValueError("Invalid archive path")
                path.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(member)
                if relative in info["files"]:
                    data = (
                        data.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")
                    )
                path.write_bytes(data)
        patch = ROOT / "third_party" / info["patch"]
        if patch.stat().st_size:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(folder),
                    "apply",
                    "--check",
                    "--whitespace=nowarn",
                    str(patch),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(folder), "apply", "--whitespace=nowarn", str(patch)],
                check=True,
            )
        for relative, digest in info["files"].items():
            if (
                hashlib.sha256(
                    (folder / relative).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                != digest
            ):
                raise ValueError(f"Patched file mismatch: {name}/{relative}")
        print(f"{name}: {folder}")
    print("Install Python dependencies in the environment used by each model.")


if __name__ == "__main__":
    main()
