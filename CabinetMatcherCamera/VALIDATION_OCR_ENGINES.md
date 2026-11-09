# ML Kit / DBNet + CRNN 验证

2026-10-29，`1.8-ocr-engines`，versionCode 10，包名 `com.example.cabinetmatcher.camera`。

## 修改范围

- 模型菜单新增 `Google ML Kit v2（中文）`（稳定 ID `ML_KIT_CHINESE`）、`DBNet + CRNN（中文）`（`DBNET_CRNN`），保留已有五种方案。七种 OCR 可搭配两种 YOLO 尺寸。
- ML Kit 使用官方随包依赖 `com.google.mlkit:text-recognition-chinese:16.0.1`，APK 中已核对中文 / 拉丁识别资源和本机动态库。使用 SDK 后端，通过 `TextEngine` 接口接入核心管线；选中时不加载 Paddle 检测和识别网络。SDK 的延迟初始化包含在首次推理耗时中，客户端由当前管线持有并复用。当前应用仍移除 INTERNET / ACCESS_NETWORK_STATE 权限。
- ML Kit 整图调用返回文字行四角，映射回原分析帧后按原几何规则匹配；整图识别字符串不参与匹配，选中裁剪另跑 SDK。SDK 实际调用范围、分数来源和 ONNX 字典字段差异在每帧 JSON 的 `ocr_engine` / `models` 中说明。ML Kit 没有独立检测框分数，不使用 Paddle 的检测框分数阈值；识别阈值仍生效。
- DBNet + CRNN 复用 v4 的 DB 系列文字检测器，识别换为 RapidOCR 官方汇总的中文 MobileNetV3 / 双向 LSTM / CTC 网络。ONNX 实际含 2 个 LSTM 节点，输入 `[batch,3,32,width]`，输出 6625 类，专用字典从 ONNX 内嵌 character 提取（6623 字符，另加 blank / 空格）。不改动原始模型文件，SHA-256 为 `6701d6db953d6e803024ec2e86ab3a5ec9132a2cf47b15b843ce6f1522486a55`。
- CRNN 上游导出元数据中的时间步写为 25，动态宽度 320 实际输出为 80；代码始终读取实际输出 TensorInfo。桌面 ONNX Runtime 和 Android 实际运行均确认可用，不以过时的输出时间步做解码。
- 原有几何匹配、方向分类、串并行汇合、过期帧抑制和导出流程继续适用。模型选择按名称持久化，兼容旧偏好。SDK 等待在协调工作线程进行，SDK 调用完成前不回收输入位图；模型切换沿用同一协调队列，避免关闭仍在使用的客户端。

## 已完成验证

- `assembleDebug`、`assembleDebugAndroidTest`、`testDebugUnitTest`、`lintDebug` 成功，见 `validation/ocr-engines/build-final.log`。67 项 JVM 测试，0 失败；lint 0 错误、24 警告。
- 全部模型 / 字典 SHA-256 与资产清单一致，CRNN 通过 ONNX checker，核对输入高度、类别和 LSTM 节点。复现脚本 `tools/fetch_crnn.py`；上游来源及形状见 `validation/ocr-engines/crnn-model.log` 和 assets 清单。
- Android 36.1 x86_64 无窗口模拟器，WHPX、4 核 / 4 GB，4 项 instrumentation 测试通过，约 59 秒。见 `validation/ocr-engines/instrumentation-final.log`：
  1. 七种模型逐个菜单切换，Activity 重建和新启动恢复最后选择。
  2. 七种模型实际运行同一柜体样图，验证文字输出、几何配对、调用数量和 JSON 中各后端的身份 / 字典字段。
  3. 新增两种后端在同图串行 / 并行下，检测框、文字和分数一致；已有低置信度过滤继续有效。
  4. 模拟器摄像头连续处理、同帧手部结果、预览保持播放、暂停及忙碌时从 ML Kit 640 切换至 CRNN 1280；旧帧未覆盖新模型结果，后台关闭相机。
- 数据见 `validation/ocr-engines/android-model-results.json`、`summary.json`；摄像头导出见 `camera-live-result.json`（ML Kit）和 `camera-screen.png`（切换后 CRNN）。菜单截图为 `model-menu.png`。

## 样图观察与限制

固定 `demo_cabinet.jpg`、YOLO 640、OCR 长边 1600、默认方向纠正与识别阈值 0.5：所有方案均得到 3 对几何匹配，均有文字输出，无逐区域识别异常。

DBNet + CRNN 两块铭牌输出“合闸/分闸转换把手”，另一个标签为“2KK49F42刀闸”，三对均通过默认阈值。ML Kit 同图有错字 / 漏字，三块裁剪的 SDK 行置信度约 0.394–0.490，均按原阈值标为 `low_confidence`，原始文字正常保留供查看。功能测试通过代表后端真实可运行，**不代表新增方法识别更准，也不代表 ML Kit 默认阈值通过**。需要时可自行降低阈值比较；分数不在统一尺度，不能按数字直接排名。

降低阈值只改变结果是否标为通过，不会修正识别出来的错字。

这只是单张图片和模拟器功能验证，未做标注数据集准确率评测；单帧耗时含首次初始化和模拟器调度，不能作为手机 FPS 或速度排名。本次未连接真实手机，未将 APK 安装到用户手机，真机功耗、发热、峰值内存及现场效果待测。

## 安装包

- `dist/CabinetMatcherCamera-mlkit-crnn-arm64-debug.apk`，355748444 字节（约 339 MiB）。
- SHA-256：`628c61abecb9969213bbd30059309f0113d46f7111576fe3bca852c36c1da10f`。
- 同包名覆盖安装，暂停识别后从顶部 OCR 菜单选择新方法，等“已就绪”再开始。对比方法见 `MODEL_SWITCH.md`。
