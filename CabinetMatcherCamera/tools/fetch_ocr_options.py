"""Fetch pinned RapidOCR ONNX models, verify upstream hashes and extract their dictionaries.

Run with a Python environment containing requests and onnx. Models stay offline in the APK.
Upstream: RapidAI/RapidOCR python/rapidocr/default_models.yaml, revision v3.10.0 models.
"""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import onnx
import requests

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app/src/main/assets"
BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/"
MODELS = [
    ("PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx", "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"),
    ("PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx", "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5"),
    ("PP-OCRv5/det/ch_PP-OCRv5_det_server.onnx", "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad"),
    ("PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx", "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a"),
    ("PP-OCRv6/det/PP-OCRv6_det_small.onnx", "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    ("PP-OCRv6/rec/PP-OCRv6_rec_small.onnx", "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
]
DICTIONARIES = {
    "ch_PP-OCRv5_rec_mobile.onnx": "rec_character_dict_v5.txt",
    "ch_PP-OCRv5_rec_server.onnx": "rec_character_dict_v5.txt",
    "PP-OCRv6_rec_small.onnx": "rec_character_dict_v6_small.txt",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(item):
    relative, expected = item
    target = ASSETS / Path(relative).name
    if not target.exists() or sha(target) != expected:
        temporary = target.with_suffix(".download")
        session = requests.Session()
        # Set OCR_DOWNLOAD_USE_PROXY=1 if the local network requires an environment proxy.
        session.trust_env = os.environ.get("OCR_DOWNLOAD_USE_PROXY") == "1"
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount("https://", adapter)
        with session.get(BASE + relative, timeout=(30, 180), stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if sha(temporary) != expected:
            raise ValueError("Upstream SHA-256 mismatch: " + relative)
        temporary.replace(target)
    print("Verified", target.name, target.stat().st_size, flush=True)
    model = onnx.load(str(target))
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    return relative, target, model, metadata


def shape(value):
    return [dim.dim_param if dim.dim_param else dim.dim_value for dim in value.type.tensor_type.shape.dim]


def main():
    # Update the manifest only after every download and dictionary validation succeeds.
    manifest_file = ASSETS / "models_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, MODELS))
    dictionaries = {}
    for relative, target, model, metadata in results:
        entry = {"url": BASE + relative, "sha256": sha(target), "bytes": target.stat().st_size,
                 "inputs": {v.name: shape(v) for v in model.graph.input},
                 "outputs": {v.name: shape(v) for v in model.graph.output}}
        if target.name in DICTIONARIES:
            if "character" not in metadata:
                raise ValueError("No embedded dictionary in " + target.name)
            characters = metadata["character"].splitlines()
            classes = shape(model.graph.output[0])[-1]
            if len(characters) + 2 != classes:
                raise ValueError(f"Dictionary mismatch: {target.name}: {len(characters)} + 2 != {classes}")
            dictionary_name = DICTIONARIES[target.name]
            content = ("\n".join(characters) + "\n").encode("utf-8")
            if dictionary_name in dictionaries and dictionaries[dictionary_name] != content:
                raise ValueError("Mobile/server dictionaries differ")
            dictionaries[dictionary_name] = content
            entry["dictionary_file"] = dictionary_name
            print(target.name, "classes", classes, "input", entry["inputs"], flush=True)
        manifest["models"][target.name] = entry
    for name, content in dictionaries.items():
        target = ASSETS / name
        target.write_bytes(content)
        manifest["models"][name] = {"sha256": sha(target), "bytes": len(content),
            "dictionary_classes_with_blank_and_space": len(content.decode("utf-8").splitlines()) + 2,
            "source": "character metadata from paired RapidOCR ONNX model"}
    manifest["ocr_versions"] = list(dict.fromkeys(
        ["V4", "V6_TINY", "V5_MOBILE", "V5_SERVER", "V6_SMALL"] + manifest.get("ocr_versions", [])))
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
