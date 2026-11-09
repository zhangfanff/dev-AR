# 第三方来源

- 1.8 新增 Google ML Kit Text Recognition v2 中文随包 SDK：`com.google.mlkit:text-recognition-chinese:16.0.1`，模型和本机动态库由官方 AAR 提供，包含中文与拉丁文字识别资源。使用 Google ML Kit SDK 条款；不可将该 SDK 或其权重当作 Apache-2.0 模型。[接入说明](https://developers.google.com/ml-kit/vision/text-recognition/v2/android)、[条款与隐私](https://developers.google.com/ml-kit/terms)。本 App 仍移除网络权限，图像只在本机处理。
- 1.8 新增中文 CRNN：RapidAI/RapidOCR 发布者 SWHL 的 `PP-OCRv1/ch_ppocr_mobile_v2.0_rec_infer.onnx`，Apache-2.0，保留现有 PaddleOCR / RapidOCR 许可。来源：[固定修订模型](https://huggingface.co/SWHL/RapidOCR/blob/38f67e0558416db6056ddd8613292d23b5ff002c/PP-OCRv1/ch_ppocr_mobile_v2.0_rec_infer.onnx)，SHA-256 `6701d6db953d6e803024ec2e86ab3a5ec9132a2cf47b15b843ce6f1522486a55`。实际含两层双向 LSTM，字典来自 ONNX 的 character 元数据；架构与预处理参照 [PaddleOCR CRNN 中文训练配置](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)。检测端复用已授权打包的 PP-OCRv4 DB 系列检测器。复现脚本 `tools/fetch_crnn.py`。

- 本版新增 MediaPipe Tasks Vision 1.0.0（配套 tasks-core，Apache-2.0），沿用独立 HandLandmarkDemo 的官方 HandLandmarker full `float16/1` 模型：https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task 。模型 SHA-256 见 `models_manifest.json`；运行使用 MediaPipe CPU，同帧输出 21 个手部关键点。官方项目：https://github.com/google-ai-edge/mediapipe 。

- PaddleOCR / PP-OCRv4：PaddlePaddle/PaddleOCR，Apache-2.0。OCR ONNX 和字典来自原项目 `rapidocr-onnxruntime==1.4.4` 的配套文件。
- PP-OCRv6 tiny：从现有 Python v6 项目复制的 RapidAI/RapidOCR 官方转换 ONNX，ModelScope 修订 v3.10.0；检测为 `onnx/PP-OCRv6/det/PP-OCRv6_det_tiny.onnx`，识别为 `onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx`。使用该模型专用 6904 字符字典，含 blank/空格后为 6906 类。来源基址：https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/ 。哈希见 `app/src/main/assets/models_manifest.json`。
- 本次新增 PP-OCRv5 mobile / server 和 PP-OCRv6 small：PaddlePaddle/PaddleOCR、RapidAI/RapidOCR，Apache-2.0；沿用 `licenses/original/PaddleOCR-LICENSE.txt`、`RapidOCR-LICENSE.txt`。从同一 ModelScope 修订 v3.10.0 下载对应 det / rec ONNX，逐文件核对 RapidOCR 官方 `python/rapidocr/default_models.yaml` 公布的 SHA-256。v5 的 18383 字符、v6 small 的 18708 字符字典来自各识别模型的 character 元数据，另加 CTC blank / 空格。原始下载 URL、哈希、输入输出形状均记录在 `models_manifest.json`，可用 `tools/fetch_ocr_options.py` 复现。
- RapidOCR：RapidAI/RapidOCR，Apache-2.0。本项目按其现有前后处理流程移植 Java 实现，没有在 Android 中嵌入 Python 包。
- Ultralytics YOLO：Ultralytics/ultralytics。当前权重导出元信息标识 AGPL-3.0；许可与来源记录沿用原项目。模型训练权重由用户提供。
- ONNX Runtime：microsoft/onnxruntime，MIT，Android AAR 1.19.2。
- OpenCV：opencv/opencv，Apache-2.0，Android AAR 4.9.0。
- AndroidX ExifInterface：Apache-2.0。Gradle Wrapper：Apache-2.0。
- 摄像头版新增 AndroidX CameraX 1.4.2 和 AndroidX Activity 1.9.3：Apache-2.0。

原项目第三方说明与许可副本保存在 `licenses/`。具体许可以相应项目原件为准。
