import argparse, sys, json, time, hashlib, platform
from pathlib import Path
import numpy as np
from PIL import Image
import torch

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["maskrcnn", "yolact", "rfdetr"], required=True)
p.add_argument("--project", required=True)
p.add_argument("--checkpoint", required=True)
args = p.parse_args()
sys.path.insert(0, args.project)
if args.model == "yolact":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dependencies import activate

    activate("yolact")
root = Path(__file__).resolve().parents[2] / "."
annotations = Path("H:/datasets/LineDatasetFull/test.json")
data = json.loads(annotations.read_text())
infos = sorted(data["images"], key=lambda i: i["id"])
images = [
    np.array(
        Image.open(
            Path("H:/datasets/LineDatasetFull/spliting_imgs/test") / i["file_name"]
        ).convert("RGB")
    )
    for i in infos
]
assert len(images) == 229
torch.set_grad_enabled(False)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
if args.model == "maskrcnn":
    from wireseg.common import load_checkpoint
    from wireseg.modeling import build_model

    c = load_checkpoint(Path(args.checkpoint))
    cfg = c["config"]
    model = build_model(cfg)
    model.load_state_dict(c["model"], strict=True)
    model = model.cuda().float().eval()

    def infer(image):
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).cuda()
        result = model([tensor])[0]
        keep = torch.argsort(result["scores"], descending=True, stable=True)
        keep = keep[result["scores"][keep] >= 0.3][:100]
        return result["boxes"][keep].cpu().numpy(), (
            result["masks"][keep, 0] >= 0.5
        ).cpu().numpy()
elif args.model == "yolact":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    from data import cfg, set_cfg
    from yolact import Yolact
    from layers.output_utils import postprocess
    from utils.augmentations import BaseTransform

    set_cfg("yolact_new700_test_config")
    cfg.mask_proto_debug = False
    model = Yolact()
    model.load_weights(args.checkpoint)
    model.cuda()
    model.float()
    model.eval()
    model.detect.use_fast_nms = False
    model.detect.use_cross_class_nms = False
    transform = BaseTransform()

    def infer(image):
        im = np.ascontiguousarray(image[:, :, ::-1])
        h, w = im.shape[:2]
        x, _, _, _ = transform(
            im,
            np.zeros((1, h, w), np.float32),
            np.array([[0.0, 0.0, 1.0, 1.0]], np.float32),
            {"labels": np.array([0]), "num_crowds": 0},
        )
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
        cls, scores, box, masks = postprocess(
            model(x), w, h, crop_masks=True, score_threshold=0.3
        )
        mask_scores = scores[1] if isinstance(scores, list) else scores
        keep = torch.argsort(mask_scores, descending=True)[:100]
        return box[keep].cpu().numpy(), masks[keep].cpu().numpy()
else:
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview(pretrain_weights=args.checkpoint)

    def infer(image):
        pred = model.predict(image, threshold=0.3)
        keep = np.argsort(-pred.confidence)[:100]
        return pred.xyxy[keep], pred.mask[keep]


with torch.inference_mode():
    for _ in range(10):
        infer(images[0])
    torch.cuda.synchronize()
    times = []
    for i, image in enumerate(images):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = infer(image)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        if i % 50 == 0 or i == 228:
            print(args.model, i + 1, 229, flush=True)
report = {
    "model": args.model,
    "checkpoint": args.checkpoint,
    "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
    "annotation_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
    "image_ids": [i["id"] for i in infos],
    "images": 229,
    "batch": 1,
    "precision": "FP32",
    "warmup_passes": 10,
    "gpu": torch.cuda.get_device_name(0),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "mean_ms": float(np.mean(times)),
    "median_ms": float(np.median(times)),
    "per_image_ms": times,
    "definition": "Input RGB already in memory; includes model-specific preprocessing, transfer, forward, postprocessing and materializing retained masks/boxes on CPU; excludes disk IO, model load, COCO scoring and RLE encoding.",
}
(root / "outputs" / f"wire_latency_{args.model}.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print("TIMING_RESULT", args.model, report["mean_ms"], flush=True)
