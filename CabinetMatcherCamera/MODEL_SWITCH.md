# 摄像头 OCR 模型对照测试版（2026-10-29）

2026-11-09 新增独立的“文字检测方向”菜单，默认单方向，可切换四方向检测并合并。所有 OCR 均支持；ML Kit 四方向模式每帧整图 SDK 调用为 4 次，单方向为 1 次，再加已匹配区域识别。使用与计时见 [FOUR_DIRECTION_TEXT.md](FOUR_DIRECTION_TEXT.md)。下文单次整图调用描述对应默认单方向模式。

项目仍在 `G:\AndroidStudioProjects\CabinetMatcherCamera`，包名不变。两个下拉菜单在应用顶部；**暂停识别 → 选择 YOLO 输入尺寸及 OCR 模型 → 等待“已就绪” → 开始识别**。摄像头可以保持预览。两项选择会记住，下次启动继续使用。

每种 OCR 均可搭配 YOLO 640 或 1280，共 14 种组合。所有模型都打包在 APK 中，在本机离线执行，首次使用无需下载。YOLO / Paddle / CRNN 使用 ONNX Runtime CPU，ML Kit 使用自己的内置 SDK 后端。

| OCR 检测与识别 | 字典类别（含 blank / 空格） | 测试用途 |
|---|---|---|
| PP-OCRv4 | 6625 | 原版对照 |
| PP-OCRv6 tiny | 6906 | 原轻量版对照 |
| PP-OCRv5 mobile | 18385 | 移动端替代方案 |
| PP-OCRv5 server | 18385 | 较大模型对照，手机上可能明显更慢、占用更多内存 |
| PP-OCRv6 small | 18710 | 比 tiny 更大的 v6 方案 |
| Google ML Kit v2（中文）（新增） | SDK 管理 | Google 中文及拉丁字符 SDK 对照 |
| DBNet + CRNN（中文）（新增） | 6625 | DB 系列检测＋中文 MobileNetV3 / BiLSTM / CTC 识别 |

server 是模型规格名称，不需要服务器。PP-OCR 选项成套选择检测和识别网络；DBNet + CRNN 刻意复用现有 v4 的 DB 系列检测器，只将识别换为经典中文 CRNN。ML Kit 采用 `com.google.mlkit:text-recognition-chinese:16.0.1` 的随包版本，SDK 自行管理网络与字典。模型大小或官方整体成绩不保证本项目现场效果更好，需要对同一柜体实测。

ML Kit 的 API 一次同时检测和识别，不能单独调用检测。因此本项目整图调用一次，仅使用文字行位置做原有几何匹配，丢弃整图文字字符串；再对选中区域进行透视裁剪、可选 0/180° 纠正和 SDK 识别。每帧 SDK 调用次数为 `1 + 选中区域数量`，这两部分耗时分别在 `ocr_detection` 和 `text_loop.recognition` 计时中；JSON 的 `ocr_engine` 也注明调用范围。SDK 不提供单独文字框置信度，本项目不对 ML Kit 文字框套用 Paddle 检测框阈值；裁剪后的识别结果仍按页面 OCR 阈值过滤。ML Kit 分数是按字符数加权的行置信度，与 CTC 分数不在统一尺度。**默认 0.5 下可能显示低置信度，原始文字仍保留在诊断预览中，可自行降低阈值对照。**

CRNN 使用 RapidOCR 发布的 `ch_ppocr_mobile_v2.0_rec_infer.onnx`（PP-OCRv1 中文轻量识别），实际含两个双向 LSTM 节点，输入高度 **32**；原有其他 ONNX 识别器仍为 48。它是传统算法对照，没有额外训练柜体专用数据。下载、校验、字典提取可用 `tools/fetch_crnn.py` 复现，原始 ONNX 不做修改。

所有组合使用的旋钮权重均是用户提供的 `G:\pythonProjects\yolov12-main\runs\detect\train9\weights\best.pt`。检查 checkpoint 的网络结构和训练参数确认它是 **YOLOv8n、单类别 `konb`、训练尺寸 640**。目录名字不决定模型版本。1280 是同一权重用更大输入推理，并未重新训练。

导出文件位于该权重旁的 `onnx\knob_640.onnx` 和 `onnx\knob_1280.onnx`，并已复制进 App assets。固定 batch=1、FP32、opset 17、无图内 NMS、无动态输入尺寸。`tools/export_train9.py` 保存复现脚本，`tools/export_manifest.json` 记录源权重哈希、输出形状和与 PyTorch 的两张样图数值对照；App 中 `models_manifest.json` 校验所有模型和字典的 SHA-256。

切换 OCR 会一起切换后端及配套模型；ONNX 方案启动时验证字典长度与网络输出，ML Kit 使用 SDK 内置字典。v5 / v6 及 CRNN 字典从已校验的 ONNX 内嵌 character 元数据提取，v5 mobile / server 使用相同字典，v6 small 使用自己的字典。`tools/fetch_ocr_options.py` 可重新下载并校验新增模型；来源是 [RapidAI/RapidOCR 官方模型配置](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml) 中固定 v3.10.0 的 ModelScope 文件。可选方向分类器沿用原模型、默认开启。模型加载和 OCR 推理共用协调线程，切换时暂停识别、取消旧帧发布，旧调用结束后才释放旧模型。连续帧复用当前主网络，仅加载当前一套组合。旧版布尔型 OCR 设置会自动迁移，新版按稳定模型 ID 保存。

## 速度该怎么看

- 顶部显示所选模型及单独的模型加载时间，加载不计入每帧处理。
- 最近完成帧旁显示**该帧实际使用的组合**；切换后尚未完成的新结果不会改变旧结果的标签。
- 显示本帧处理时间、收到分析帧到界面提交的时间、平均结果更新帧率。更新帧率不是相机预览的 FPS；曝光、驱动等待和屏幕实际绘制不在计时内。
- 标记模型加载后的首次推理；每次点击开始识别，另统计去除本次首帧后的处理、YOLO、OCR 平均时间。暂停后重新开始会清空本次均值。
- 详细计时包含帧格式转换、YOLO 预处理/网络/NMS、文字检测预处理/网络/后处理、几何匹配、已匹配文字的裁剪/识别/解码、绘图。父合计已含子步骤，不能重复累加。
- JSON 新增 `models`、`parameters.yolo_input_size`、`parameters.ocr_model`；保存模型文件名、字典、CPU 后端、加载时间及首次推理标记，便于按组合统计。原来所有匹配/未匹配、候选及坐标仍保留。

比较时对准同一个柜子、保持相同距离和画面，固定**视频分析分辨率、OCR 检测长边、匹配阈值、方向分类设置**，每组连续观察数帧再读均值。YOLO 尺寸、视频采集尺寸、OCR 检测尺寸是三项独立设置。1280 输入通常更耗时；640 可能漏掉小旋钮。不同 OCR 可能产生不同区域数量，导致匹配数量及逐框识别调用次数变化，因此同时查看计时和检测/匹配结果。固定图片的对照结果见 `validation/model-switch/`，不能把图片管线耗时直接当摄像头 FPS。

建议先依次比较 v4、v5 mobile、v6 small，再尝试 v5 server。暂停后在匹配区域原图预览中核对选中的位置、OCR 原始文字和分数，导出每组 JSON / PNG。不同模型的置信度不是统一校准后的准确率，不能只按分数高低判断优劣。只有框选对了但文字读错时，才是逐区域识别的问题；文字区域漏检和旋钮配错需分别观察。

1.2 版显示更新见 `LIVE_OVERLAY.md`：匹配结果同时叠加在实时预览和下方历史帧。帧策略保持 CameraX `KEEP_ONLY_LATEST`、最多一个推理，每处理完一帧再接收最新帧，不积压每个视频帧。

## 保留与恢复

历史验证见 `VALIDATION_MODEL_SWITCH.md`、`VALIDATION_OCR_OPTIONS.md`。本次 1.8 版验证见 `VALIDATION_OCR_ENGINES.md`。

## 构建

在 Android Studio 打开项目，Gradle JDK 使用 `D:\Android Studio\jbr`，SDK 使用 `G:\Android\Sdk`。构建命令仍见 README。手机安装 `app/build/outputs/apk/debug/app-arm64-v8a-debug.apk`；同包名更新安装，保留相机权限和原应用数据。
