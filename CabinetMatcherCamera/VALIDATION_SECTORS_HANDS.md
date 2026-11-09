# 扇区匹配与同帧手部检测验证

2026-10-19；版本 `1.3-camera-sectors-hands`，versionCode 4，包名 `com.example.cabinetmatcher.camera`。

## 验证结果

- Debug APK、测试 APK、JVM 测试和 lint 均成功，见 `validation/sectors-hands-build.log`、`validation/sectors-hands-test-build.log`。最后细化计时标签后另执行 APK/lint 构建，见 `validation/sectors-hands-final-build.log`。lint 为 0 错误、24 项警告，主要为界面文字和现有 API 提示。
- **41 项 JVM 单元测试通过**。在原 33 项测试基础上增加 8 项：四个方向分别选择正确标签、四向按全局距离而非轴次序挑选、置信度优先及一对一占用、扇区边界、侧向限制随方向旋转、四向窄扇区的对角空隙和同中心拒绝、默认开启文字方向判断、无效方向拒绝。原向下模式的侧面文本拒绝测试保留并通过。
- **8 项 Android 模拟器测试通过**，见 `validation/sectors-hands-emulator-tests.log`：
  - 3 项集成测试：v4 与 v6 tiny 分别在真实柜体样图上运行默认方向分类和 OCR，并在同进程、同图运行手部模型；官方手图真实 21 点结果的绘制与同帧 JSON；非向下模式的参数、匹配轴和候选拒绝原因。
  - 4 项原叠加测试：裁剪/四种旋转、预览黑边映射、矢量绘制/清除、JSON 预览策略。
  - 1 项摄像头完整流程测试：慢推理时预览继续、连续两帧相机时间戳新鲜、每帧柜体和手部均已完成才发布、手部/视频帧号与原始相机时间戳一致、默认方向判断开启、暂停和模型切换不发布旧帧、后台清除叠加并关闭相机。
- 集成手图 `validation/sectors-hands/combined-hand-example.png` 已检查：两只手的 21 点骨架与画面对应，两个食指指尖用黄色标出。相关 JSON 与摄像头结果一并保存在该目录。
- MediaPipe 与 ONNX Runtime、OpenCV 可在此合成摄像头和静态样图测试中共存；正式循环在同一分析线程串行调用，不另建手部视频帧队列。手模型与全部 ONNX/字典已纳入资产 SHA-256 校验。

## 测试边界与交付

此次手机没有连接，没有执行真机安装、相机或速度测试。模拟器使用合成摄像头，静态手图来自 MediaPipe 官方测试资源；这些结果不等于真实柜体、手部运动与光照下的准确率或手机帧率。摄像头测试中的 2 秒人工延迟仅为测试参数，正式 App 默认为 0。

新版 ARM64 APK 与源码 zip 使用 `CabinetMatcherCamera-sectors-hands-*` 名称。原 1.2 版 APK 与完整源码 zip 保留为 `CabinetMatcherCamera-live-overlay-*`。原相册工程、独立 HandLandmarkDemo 和 Python 项目没有修改。

0/180° 方向分类纠正的能力范围、四向半角 45° 的完整覆盖以及各耗时键含义见 `SECTORS_AND_HANDS.md`。
