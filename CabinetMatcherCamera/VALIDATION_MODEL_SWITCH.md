# 模型切换版验证

日期：2026-10-19。版本 `1.1-camera-model-switch` / versionCode 2；包名 `com.example.cabinetmatcher.camera`。

## 已完成

- 新权重源文件 SHA-256：`6c2e11626e65247694de5a0c458810fb6c6880eaa69fd22d7a7f31d1bbcf45ac`，架构为 YOLOv8n，单类别 `konb`。源 `.pt` 未修改。
- 640 导出输入 `[1,3,640,640]`、输出 `[1,5,8400]`；1280 导出输入 `[1,3,1280,1280]`、输出 `[1,5,33600]`。两者均通过 ONNX checker，并使用 ONNX Runtime CPU 与 PyTorch 在 `1.jpg`、`2.jpg` 上对照原始预测。最大置信度绝对误差小于 `1.4e-6`，置信度至少 0.25 的预测框最大坐标误差为 `0.0000611` 输入像素。具体哈希和逐图数据见 `tools/export_manifest.json`。
- 已读取并检查全部 ONNX 的输入/输出形状及两个字典长度；v4 6625 类，v6 tiny 6906 类，包含 blank 与空格。所有 App 模型/字典均纳入资产 SHA-256 校验。
- `:app:assembleDebug`、`:app:assembleDebugAndroidTest`、`:app:testDebugUnitTest`、`:app:lintDebug` 成功。33 项单元测试，0 失败；lint 0 错误，21 项警告（主要为界面文字和现有 API 提示）。日志在 `validation/model-switch-build.log`、`validation/model-switch-test-build.log`。
- 新增测试覆盖四种模型身份与字典配套、错误输入拒绝，以及 8400/33600 两种 YOLO 输出按各自缩放/灰边还原到相同原图坐标。原匹配、识别选择、帧并发/取消和 YUV 测试均保留。
- ARM64 APK 已通过 adb 更新安装到已连接的 PHB110（序列号 59509744），包管理器确认 versionCode=2。旧项目备份在 `G:\AndroidStudioProjects\CabinetMatcherCamera-backup-20261019`。

## 未执行的真机测试

测试辅助 APK 的安装被手机系统拦截；用户选择“暂时不安装测试包，只交付新版 App”，因此停止该安装，未执行四组合 Android instrumentation 测试，也没有本次真机四组合耗时或摄像头流畅度测量数据。**构建/单元测试通过不等于四种组合均已在真机验证。** 不应沿用原版本或桌面 Python 的耗时作为本版手机成绩。

`CameraSmokeTest` 已准备好四组合、两张样图、逐组多次推理、模型参数 JSON、切换期间旧帧取消、预览与推理并行、后台关闭相机等测试。之后若允许安装测试包，可在手机授权相机后运行 `:app:connectedDebugAndroidTest`；成功后应用内部的 `model-matrix-results.json` 可用于统计固定图片管线耗时（仍不等于相机结果更新 FPS）。

## 用户测试

启动新版应用，暂停识别后在顶部选择 640/1280 和 OCR v4/v6 tiny，等待加载完成后开始。保持柜子、取景、视频分辨率、OCR 检测尺寸和阈值相同，对比每帧耗时、去除首帧的均值及平均结果更新率，并核对漏检/识别/匹配结果。完整说明见 `MODEL_SWITCH.md`。
