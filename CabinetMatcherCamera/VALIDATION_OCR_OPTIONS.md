# OCR 模型扩展验证

日期：2026-10-27。版本 `1.7-ocr-options`，versionCode 9，包名 `com.example.cabinetmatcher.camera`。

## 实现

- 保留 PP-OCRv4 / PP-OCRv6 tiny，新增 PP-OCRv5 mobile、PP-OCRv5 server、PP-OCRv6 small。检测和识别各用对应模型，共 5 种 OCR，可搭配现有 640 / 1280 YOLO。
- 新增六个 ONNX 文件均来自 RapidAI/RapidOCR 的 ModelScope 固定修订 v3.10.0，SHA-256 与官方模型配置一致。字典从配套 ONNX 元数据提取，v5 mobile / server 共用 18385 类字典，v6 small 为 18710 类，均含 CTC blank / 空格。
- 模型菜单从枚举生成，保存稳定名称，兼容旧版 `ocrV6` 布尔偏好；进程重启和 Activity 重建均恢复所选模型。原图裁剪、方向分类、匹配、识别阈值、串并行调度和计时口径沿用现有实现。
- Android 无网络权限，所有模型均随 APK 打包，server 也是本机 ONNX Runtime CPU 推理。

## 已完成验证

1. 全部资产 SHA-256 与清单一致；新增六个 ONNX 均通过 ONNX checker，输入输出形状和字典长度匹配。见 `validation/ocr-options/assets-check.json`。
2. `assembleDebug`、`assembleDebugAndroidTest`、`testDebugUnitTest`、`lintDebug` 成功。66 项 JVM 测试，0 失败；lint 0 错误、24 警告。见 `validation/ocr-options/build.log` 和 `app/build/reports/lint-results-debug.txt`。
3. Android 36.1 x86_64 无窗口模拟器，WHPX，4 核 / 4 GB 内存，运行 `OcrOptionsTest`，2 项测试通过，约 60 秒。见 `validation/ocr-options/instrumentation.log`。
4. 菜单测试依次加载全部五种 OCR，验证切换后的实际模型身份，再验证 Activity 重建和全新启动恢复 v6 small。
5. 图片测试在 Android ONNX Runtime 1.19.2 下逐个实际运行全部五套模型，使用 `demo_cabinet.jpg`、YOLO 640、OCR 长边 1600、默认方向纠正。每套均得到 3 对几何匹配和 3 次成功的文字识别，未发生逐区域识别异常；导出模型文件名、字典、类别数均正确。见 `validation/ocr-options/android-model-results.json`、`summary.json`。
6. 菜单界面截图见 `validation/ocr-options/model-menu.png`，已检查五个选项显示完整。

同图输出中，两块铭牌均为“合闸/分闸转换把手”；另一块 v4 / v5 输出“2KK49F42刀闸”，v6 输出“2KK 49F42刀闸”。这张样图未证明新增模型准确率更高，只确认所有方案在当前管线中可运行并输出中文。每套只跑一帧，初始化、首次推理和模拟器调度会影响耗时，**不得将该记录作为手机速度排名或摄像头 FPS**。

## 安装包与限制

- 手机 ARM64 安装包：`dist/CabinetMatcherCamera-ocr-options-arm64-debug.apk`，337206284 字节（约 322 MiB）。新增 server 模型占用较多空间，因此 APK 比旧版大。
- SHA-256：`29168f368e0a543629dfd56bcba9280b973286188c92c330c6ff2b3450ae7f51`。
- 也已构建 x86_64 和 universal 调试 APK，位于 `app/build/outputs/apk/debug/`。
- 本次未连接真实手机，没有安装到用户手机，未测量真机发热、峰值内存、摄像头实时性能和现场识别率。测试方法见 `MODEL_SWITCH.md`；请固定取景、尺寸、执行模式与阈值，观察多帧结果与去除首帧后的平均耗时。
- 模拟器测试和构建不会替代现场评估；较大模型不保证在模糊、反光、倾斜或低分辨率图像上识别更准。
