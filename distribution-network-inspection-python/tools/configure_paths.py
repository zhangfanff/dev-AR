"""Replace configured path prefixes only inside this copied repository."""

from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply and args.check:
        parser.error("Choose --check or --apply")
    root = Path(__file__).resolve().parents[1]
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v
        for k, v in mapping.items()
    ):
        parser.error("Mapping must contain nonempty string paths")
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix not in {".py", ".json", ".yaml", ".yml"}:
            continue
        if any(
            part in {"vendor", "ultralytics", "docs", ".git", "__pycache__"}
            for part in file.relative_to(root).parts
        ):
            continue
        if file.name.startswith("local_paths") or file == Path(__file__).resolve():
            continue
        original = file.read_text(encoding="utf-8-sig")
        updated = original
        for old, new in mapping.items():
            updated = updated.replace(old, new.replace("\\", "/"))
        if updated != original:
            print(file.relative_to(root))
            if args.apply:
                file.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
