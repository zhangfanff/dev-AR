"""Fetch the Chinese MobileNetV3/BiLSTM/CTC CRNN published by RapidOCR.

Requires requests + onnx. The default mirror serves the same pinned Hugging Face object;
the upstream LFS SHA-256 is checked before the model is used.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import onnx
import requests

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app/src/main/assets"
REVISION = "38f67e0558416db6056ddd8613292d23b5ff002c"
RELATIVE = f"SWHL/RapidOCR/resolve/{REVISION}/PP-OCRv1/ch_ppocr_mobile_v2.0_rec_infer.onnx"
EXPECTED = "6701d6db953d6e803024ec2e86ab3a5ec9132a2cf47b15b843ce6f1522486a55"


def main():
    target = ASSETS / "ch_ppocr_mobile_v2.0_rec_infer.onnx"
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != EXPECTED:
        session = requests.Session()
        session.trust_env = False
        temporary = target.with_suffix(".download")
        with session.get("https://hf-mirror.com/" + RELATIVE + "?download=true", stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED:
            raise ValueError("CRNN SHA-256 mismatch")
        temporary.replace(target)
    model = onnx.load(str(target))
    onnx.checker.check_model(model)
    operations = Counter(node.op_type for node in model.graph.node)
    if not operations["LSTM"]:
        raise ValueError("Expected a genuine recurrent CRNN (LSTM operators)")
    metadata = {x.key: x.value for x in model.metadata_props}
    # Older RapidOCR exports have no dictionary metadata; use the original Paddle character list.
    if "character" in metadata:
        dictionary = ("\n".join(metadata["character"].splitlines()) + "\n").encode("utf-8")
        dictionary_source = "ONNX character metadata"
    else:
        # ppocr_keys_v1.txt, already bundled for v4, has the same original Chinese CTC ordering.
        dictionary = (ASSETS / "rec_character_dict.txt").read_bytes()
        dictionary_source = "PaddleOCR ppocr_keys_v1.txt (same ordering as existing v4 dictionary)"
    classes = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    if len(dictionary.decode("utf-8").splitlines()) + 2 != classes:
        raise ValueError("CRNN dictionary/output mismatch")
    dictionary_file = ASSETS / "rec_character_dict_crnn.txt"
    dictionary_file.write_bytes(dictionary)
    manifest_file = ASSETS / "models_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    def shape(value):
        return [x.dim_param or x.dim_value for x in value.type.tensor_type.shape.dim]
    manifest["models"][target.name] = {
        "url": "https://huggingface.co/" + RELATIVE, "download_mirror": "https://hf-mirror.com/" + RELATIVE,
        "sha256": EXPECTED, "bytes": target.stat().st_size,
        "inputs": {x.name: shape(x) for x in model.graph.input},
        "outputs": {x.name: shape(x) for x in model.graph.output},
        "architecture": "CRNN: MobileNetV3 + bidirectional LSTM + CTC",
        "lstm_nodes": operations["LSTM"], "dictionary_file": dictionary_file.name,
    }
    manifest["models"][dictionary_file.name] = {
        "sha256": hashlib.sha256(dictionary).hexdigest(), "bytes": len(dictionary),
        "dictionary_classes_with_blank_and_space": classes, "source": dictionary_source,
    }
    manifest["ocr_versions"] = ["V4", "V6_TINY", "V5_MOBILE", "V5_SERVER", "V6_SMALL", "ML_KIT_CHINESE", "DBNET_CRNN"]
    manifest["ml_kit"] = {"artifact": "com.google.mlkit:text-recognition-chinese:16.0.1",
                          "delivery": "bundled in APK", "dictionary": "SDK-managed"}
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["models"][target.name], indent=2), flush=True)


if __name__ == "__main__":
    main()
