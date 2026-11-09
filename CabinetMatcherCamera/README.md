# 开关柜摄像头实验版

当前 **1.10-frame-text-size**（2026-11-09）：下方设置新增“文字检测尺寸”，可选择跟随相机实际分析帧，跳过手动长边缩小限制；保留默认的手动模式。单方向/四方向均适用，逐次输入尺寸会显示并导出。见 [FRAME_TEXT_SIZE.md](FRAME_TEXT_SIZE.md)。

当前 **1.9-four-direction-text**（2026-11-09）：OCR 模型下方新增“文字检测方向”下拉框，默认保留单方向检测，可选择 0°/90°/180°/270° 四方向检测并合并结果。坐标还原后沿用原匹配方向与逐区域识别，选择会保存。详细说明见 [FOUR_DIRECTION_TEXT.md](FOUR_DIRECTION_TEXT.md)。

当前 **1.6-knob-relative-matching**：每个旋钮独立以自身 YOLO 框的对角线为最大匹配距离，文本区域至少三分之一面积在该旋钮框外。面积规则替代原文本中心必须框外的限制；倾斜文字按实际四边形计算交集。页面移除旧的图像对角线比例和图宽侧向限制输入，保留方向、扇区、模型和置信度调节。匹配预览及 JSON 显示每个旋钮的距离阈值、候选框外面积比例。详见 [KNOB_RELATIVE_MATCHING.md](KNOB_RELATIVE_MATCHING.md)。

当前 **1.8-ocr-engines**（2026-10-29）：OCR 菜单新增 **Google ML Kit v2（中文）** 和 **DBNet + CRNN（中文）**，共七种方案，保留已有五种 PP-OCR。ML Kit 使用随 APK 打包的中文 SDK，DBNet + CRNN 复用 v4 的 DB 系列检测器并接入真实 MobileNetV3 / BiLSTM / CTC 中文 CRNN。两者均本机离线运行，切换后自动保存选择。暂停识别后选择，等“已就绪”再开始。对比方法和 ML Kit 计时、分数差异见 [MODEL_SWITCH.md](MODEL_SWITCH.md)，验证见 [VALIDATION_OCR_ENGINES.md](VALIDATION_OCR_ENGINES.md)。

**1.7-ocr-options**（2026-10-27）：新增 PP-OCRv5 mobile、PP-OCRv5 server、PP-OCRv6 small。历史验证见 [VALIDATION_OCR_OPTIONS.md](VALIDATION_OCR_OPTIONS.md)。

**1.5-match-region-debug**：预览中显示全部文字检测区域，明确标注 `K旋钮 → R文字框`、OCR 状态和分数；最近完成帧下方增加匹配区域的原图放大预览。低置信度、空输出及识别异常均保留区域。可暂停识别固定结果，查看每个旋钮实际选中了哪块区域，以及未匹配时的候选拒绝原因。详见 [MATCH_REGION_DEBUG.md](MATCH_REGION_DEBUG.md)。

最新 **1.4-same-frame-parallel**：顶部新增串行/并行选择；并行时同一帧的 YOLO、文字检测、手部关键点同时启动，等待旋钮与文字框后匹配并识别，等待所有分支后统一发布。两种模式都只处理一帧，保留最新帧策略。使用与计时口径见 [PARALLEL_INFERENCE.md](PARALLEL_INFERENCE.md)。下文“分析器使用单线程执行完整流程”描述的是旧串行调度；新版分析器负责 OCR 和汇合，另有 YOLO、手部固定工作线程。

最新 **1.3.1-outside-knob-center**：匹配时强制文本中心位于当前旋钮包围框外，框内和边界均排除。规则及验证见 [OUTSIDE_CENTER.md](OUTSIDE_CENTER.md)。

最新 **1.3-camera-sectors-hands**：增加上/下/左/右及四向同时的扇形匹配，默认开启 0/180° 文字方向纠正，并在同一帧完成柜体识别与手部关键点检测后处理最新帧。见 [SECTORS_AND_HANDS.md](SECTORS_AND_HANDS.md) 和 [VALIDATION_SECTORS_HANDS.md](VALIDATION_SECTORS_HANDS.md)。以下旧版本说明中的“仅向下”和“方向默认关闭”已由此版本设置替代。

2026-10-19 最新版本 **1.2-camera-live-overlay**：匹配框、文字和连线直接叠加在实时预览，显示结果年龄和更新频率；移动相机可观察处理滞后。见 [LIVE_OVERLAY.md](LIVE_OVERLAY.md)。下文原帧策略与模型选择仍适用。

项目：`G:\AndroidStudioProjects\CabinetMatcherCamera`。应用名：**开关柜摄像头**；包名：`com.example.cabinetmatcher.camera`。可与“开关柜匹配”和“开关柜先匹配”同时安装。

此项目从 CPU 先匹配版复制而来，使用 CameraX 后置摄像头预览与分析帧入口，不录制视频，也不上传图像。2026-10-19 更新：使用 train9 新权重，支持 YOLO 640/1280 和 PP-OCRv4/PP-OCRv6 tiny 四种组合，保留下方扇区一对一匹配和逐步计时。模型切换说明见 [MODEL_SWITCH.md](MODEL_SWITCH.md)，旧摄像头项目已备份。

## 使用

1. 在 Android Studio 打开本文件所在项目，等待 Gradle 同步。SDK 路径为 `G:\Android\Sdk`，Gradle JDK 选择 Android Studio 内置 JBR（Java 17 或以上）。
2. 安装 `app/build/outputs/apk/debug/app-arm64-v8a-debug.apk` 到手机。
3. 打开应用，在顶部选择 YOLO 尺寸和 OCR 模型，等待“已就绪”，点击“开启摄像头”，允许相机权限。
4. 对准柜子，点击“开始识别”。预览画面可点击对焦。
5. 上方是实时预览与最新完成结果的叠加框、文本和连线；角落显示帧号、结果年龄和更新率。下方显示该结果对应的原视频帧与完整标注。
6. 点击“暂停识别”后可调整 YOLO 输入尺寸、OCR 模型、分析帧分辨率、OCR 检测尺寸和匹配阈值。切换模型须等加载完成，再点击开始识别；分析帧分辨率变更会重新打开预览。
7. 可导出最近完成帧的完整 JSON 和 PNG。导出会暂停识别，系统文件选择器打开时摄像头关闭；返回后可重新开启摄像头。

## 帧处理策略

```text
摄像头持续输出 ──→ CameraX Preview（持续播放）
        └──────→ ImageAnalysis（KEEP_ONLY_LATEST）
                     ↓ 一次处理一帧
              YUV → BGR → 方向纠正
                     ↓
             整图文本检测 → 旋钮检测
                     ↓
             按几何坐标一对一匹配
                     ↓
             只识别匹配上的文本区域
                     ↓
             在对应帧上标注、更新结果
                     ↓
             关闭当前 ImageProxy
                     ↓
             接收下一张最新可用帧
```

`ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST` 会丢弃推理期间积压的旧帧。分析器使用单线程执行完整流程；当前 `ImageProxy` 在推理、绘制完成或失败后由 `finally` 关闭。不要提前关闭并把推理任务逐帧提交到另一条队列，否则仍可能处理排队的旧画面。

附加 `FrameGate` 保证最多一帧在处理（同帧模型可并行）；暂停、关闭摄像头或进入后台会使当前会话失效，已经开始的 ONNX 调用可以完成，但结果不再发布。下一次开始的新会话不会与旧帧并发。后台关闭相机，返回前台需要用户重新开启。

若完整处理一帧约 2 秒，结果更新率约为 **0.5 帧/秒**，摄像头预览仍可以流畅播放。这是视频流的抽帧分析，没有变成逐帧实时识别。丢帧避免队列越积越长，并不能消除当前帧本身的计算延迟。

本版同时在实时预览和对应历史帧上显示检测框。实时预览采用最新完成的旧结果，不进行运动跟踪或延迟补偿；移动相机时的框滞后就是待观察的推理延迟。

CameraX 官方说明：<https://developer.android.com/media/camera/camerax/analyze>

## 尺寸与模型

- 视频分析优先请求 1920×1080；可选择 1280×720、640×480。摄像头支持的尺寸组合决定最终分辨率，应用显示方向纠正后的实际处理尺寸；不能保证每个设备都提供请求尺寸。
- CameraX 帧可能含带行填充、像素步长为 1 或 2 的 YUV 平面；转换代码处理这些情况，也处理裁剪范围及 0/90/180/270 度方向。
- 使用后置摄像头，不镜像分析帧；应用固定竖屏，预览保持宽高比。
- YOLO 可选择固定 640×640 或 1280×1280，均来自 train9 的 YOLOv8n 新权重，等比例缩放加灰边。视频分辨率与 YOLO 输入尺寸独立。
- OCR 检测长边默认 1600，尺寸取 32 的倍数；逐区域识别高度 48、宽度至少 320。方向分类默认关闭。
- CPU ONNX Runtime，算子内线程 2、算子间线程 1，不使用 NNAPI。
- 匹配仍是旋钮置信度优先、一对一、默认正下方左右各 45°、距离不超过处理图对角线的 12%。

## 计时与结果坐标

网络逐步计时沿用 `TimingReport`：YOLO 预处理/推理/NMS、文本检测预处理/推理/后处理、匹配、逐区域裁剪/识别/解码等。增加 `camera.convert` 和 `camera.processing`，界面显示处理帧总耗时及平均结果更新率。

模型在开始识别前加载并连续复用，帧耗时不含初次模型加载。延迟从分析器收到帧开始，到主线程提交结果控件为止，不包括相机曝光、驱动内部等待和屏幕实际绘帧。`camera_timestamp_ns` 仅保存相机时间戳，不与应用时间直接相减，以免混用设备时间基准。

JSON 保留所有候选、匹配、未匹配区域与未识别原因；新增 `video`，含帧号、会话号、相机时间戳、旋转角、原始分析帧尺寸、处理耗时和丢帧策略。坐标以**方向纠正且按 CameraX cropRect 裁剪后的分析帧**为基准，与下方示意图一致，不代表相册照片或另一次拍照的坐标。

## 构建与测试

```powershell
cd G:\AndroidStudioProjects\CabinetMatcherCamera
$env:JAVA_HOME = 'D:\Android Studio\jbr'
.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest :app:lintDebug
```

原匹配与识别选择规则测试保留；新增帧忙碌丢弃、并发互斥、暂停/重启失效及 YUV 平面排列测试。设备测试覆盖方向/颜色转换、真实模型的示例配对、摄像头连续两帧、暂停后不发布旧结果与进入后台关闭预览。

设备测试需先授权相机权限：

```powershell
G:\Android\Sdk\platform-tools\adb.exe shell pm grant com.example.cabinetmatcher.camera android.permission.CAMERA
.\gradlew.bat :app:connectedDebugAndroidTest
```

最新预览叠加验证见 `VALIDATION_LIVE_OVERLAY.md`，模型切换验证见 `VALIDATION_MODEL_SWITCH.md`；`VALIDATION.md` 是更早的历史记录。相机视频的曝光、对焦、运动模糊和分辨率可能改变模型检测结果，需要用真实柜子进行测试。
