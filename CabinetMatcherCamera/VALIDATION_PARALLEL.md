# 1.4 同帧并行验证记录

日期：2026-10-19。版本 versionCode 6 / `1.4-same-frame-parallel`。

## 构建与单元测试

JDK：`D:\Android Studio\jbr`；SDK：`G:\Android\Sdk`；Gradle 离线执行：

```powershell
./gradlew.bat :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest :app:lintDebug --offline --console=plain
```

构建成功。53 项 JVM 测试通过，含新增的 4 项并发资源生命周期测试：中断后仍等原生任务完成、单分支失败后等待其他分支、清理保留中断标记、原生 Error 传播。原有匹配、扇区、中心点排除、选择性识别、FrameGate 与 YUV 测试均通过。Lint 为 0 errors / 24 warnings（现有本地化等警告）。

## Android 模拟器

使用本地 `CabinetCameraTest`，Android 36.1 Google Play x86_64，模拟后置摄像头；测试包仅安装在本地模拟器，未安装到手机。

环境限制：模拟器曾出现 Pixel Launcher / System UI 的系统 ANR 提示，原始 `camera-screen.png` 截图被系统弹窗遮挡；并非本 App 的 ANR 对话框。关闭/等待系统提示后，已复查 `parallel-clean-ui.png`，执行模式选择和 App 界面显示正常。因此不将此模拟器的耗时视为手机性能依据。

最终 11 项测试全部通过，运行约 130.6 秒：

- ParallelInferenceTest：2 项。覆盖 640/1280 × PP-OCRv4/PP-OCRv6 tiny 四种组合，每组合串行和并行各运行两次（共 16 次完整柜体与手部处理）；同一图像的旋钮框及置信度、文字框、匹配 ID、识别文字、识别状态和置信度一致；每种配置均有有效旋钮、文字和匹配，识别无错误。另验证实际相机串行→并行→串行、同帧汇合、暂停旧结果失效和最新帧策略。
- CombinedFeaturesTest：3 项。两个 OCR 版本默认方向判断，真实手部 21 点和食指坐标导出，非下方向候选拒绝原因。
- OverlayPreviewTest：4 项。实时预览坐标和绘制等既有覆盖继续通过。
- CameraSmokeTest：2 项。并行模式下连续两次慢推理、预览继续播放、下一帧为最新画面；暂停期间切换两种模型、清除旧叠加框、等旧帧退出再加载新模型、恢复后的模型元数据正确；关闭相机旧结果不再发布。另验证 90/270° 旋转像素位置与 BGR 转 Bitmap 色彩。

## 计时数据与范围

`validation/parallel/parallel-model-comparison.json` 保留全部 16 次模型验证的模拟器计时；每模式首张标记为 warmup。该测试用于功能一致性，并非严谨性能基准：每配置只有一个后续样本、顺序和宿主机负载等因素未控制，且不含正式相机路径的全部转换/绘制。不能据此推算手机加速比。

`parallel-camera-result.json` 是相机切换测试导出的并行帧，`camera-live-result.json` 与 PNG 是慢推理及切换模型测试的结果。JSON 包含 mode、max_concurrent_frames=1、并行 max_concurrent_model_tasks=3、同帧的手部 frame_id/camera_timestamp_ns，以及 detection.wall/combined.total 等耗时。

手机在本次工作中未连接，未测量手机并行收益，也未覆盖安装到手机。本地构建提供 ARM64 APK，用户可在同一 App 中切换对照。

## 交付校验

打包脚本检查 versionCode、ARM64 架构、无网络权限、APK 内 models_manifest.json 所列全部模型及字典的 SHA-256，并确认单元测试和 Android 测试日志成功后才复制交付。模型保持原有权重和 CPU 后端。

交付文件 `CabinetMatcherCamera-parallel-arm64-debug.apk`、`CabinetMatcherCamera-parallel-source.zip` 与 `CabinetMatcherCamera-parallel-delivery.json` 位于 `C:\Users\13291\Documents\Codex\2026-10-08\new-chat-2\outputs`。旧 `outside-center` 交付文件保留。
