"""Export the exact train9 checkpoint without writing over its training artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

SOURCE = Path(r'G:\pythonProjects\yolov12-main\runs\detect\train9\weights\best.pt')
OUT = SOURCE.parent / 'onnx'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'source': str(SOURCE), 'source_pt_sha256': sha(SOURCE),
            'architecture': 'YOLOv8n', 'classes': {0: 'konb'}, 'exports': {}}
for side in (640, 1280):
    work = OUT / f'export_{side}'
    work.mkdir(exist_ok=True)
    checkpoint = work / 'best.pt'
    shutil.copy2(SOURCE, checkpoint)
    model = YOLO(str(checkpoint))
    exported = Path(model.export(format='onnx', imgsz=side, batch=1, dynamic=False,
                               half=False, simplify=False, opset=17, device='cpu', nms=False))
    target = OUT / f'knob_{side}.onnx'
    shutil.copy2(exported, target)
    graph = onnx.load(str(target))
    onnx.checker.check_model(graph)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(target), sess_options=opts, providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == [1, 3, side, side]
    checks = []
    reference = YOLO(str(SOURCE)).model.eval().float()
    for name in ('1.jpg', '2.jpg'):
        image = cv2.imread(str(Path(r'G:\pythonProjects\ocr\images') / name))
        scale = min(side / image.shape[1], side / image.shape[0])
        w, h = round(image.shape[1] * scale), round(image.shape[0] * scale)
        left, top = round((side - w) / 2 - .1), round((side - h) / 2 - .1)
        resized = cv2.resize(image, (w, h))
        padded = cv2.copyMakeBorder(resized, top, side-h-top, left, side-w-left, cv2.BORDER_CONSTANT, value=(114,114,114))
        array = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None], dtype=np.float32) / 255
        with torch.inference_mode():
            expected = reference(torch.from_numpy(array))[0].numpy()
        actual = session.run(None, {session.get_inputs()[0].name: array})[0]
        # Only compare box coordinates where either backend predicts an object.
        # Empty-cell coordinates are numerically noisy and are discarded by NMS.
        selected = np.maximum(expected[:,4:,:], actual[:,4:,:]).max(axis=1) >= .25
        score_error = float(np.abs(expected[:,4:,:] - actual[:,4:,:]).max())
        box_error = float(np.abs(expected[:,:4,:] - actual[:,:4,:]).transpose(0,2,1)[selected].max()) if selected.any() else 0.
        assert score_error < .001, score_error
        assert box_error < .1, box_error
        checks.append({'image':name, 'max_score_error':score_error, 'max_box_error_pixels_above_conf_025':box_error})
    manifest['exports'][target.name] = {'sha256':sha(target), 'bytes':target.stat().st_size,
        'input':session.get_inputs()[0].shape, 'output':session.get_outputs()[0].shape,
        'opset':17,'dynamic':False,'precision':'FP32','nms_in_graph':False,'verification':checks}
    print(json.dumps(manifest['exports'][target.name], ensure_ascii=False), flush=True)
    del reference, session, graph, model
(OUT / 'export_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('DONE', OUT, flush=True)
