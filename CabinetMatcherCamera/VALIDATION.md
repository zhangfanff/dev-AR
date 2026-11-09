# 摄像头实验版验证

> 这是模型切换改动前的历史记录。2026-10-19 新权重与四组合版本请查看 `VALIDATION_MODEL_SWITCH.md`。

完成日期：2026-10-15（Asia/Shanghai）。项目：`G:\AndroidStudioProjects\CabinetMatcherCamera`。

## 构建

Android Gradle Plugin 8.8.1，compile/target SDK 35，min SDK 26，Android Studio 内置 JBR。以下任务成功：

```text
:app:assembleDebug
:app:assembleDebugAndroidTest
:app:testDebugUnitTest
:app:lintDebug
```

生成 arm64-v8a、x86_64 和通用 debug APK。Lint 无错误，有国际化、竖屏限制等警告；本实验版使用中文界面，未关闭 Lint 检查。

## 本地测试

29 项通过：原匹配 8 项、下方扇区 9 项、识别选择 4 项、帧调度 4 项、YUV 打包 4 项。

帧调度测试覆盖暂停不接收任务、1000 个忙碌时刻的帧不排队、停止/重启使旧结果失效但不重叠推理、8 个并发调用只准入一个任务。YUV 测试覆盖三平面、交错色度、行填充、非零缓冲区偏移、只读平面、VU 顺序和输入尺寸检查。

## 安卓运行测试

使用已有 SDK 镜像创建临时 **Android 36.1 x86_64 模拟器**，WHPX，模拟后置摄像头。3 项仪器测试通过：

1. 视频帧 90/270 度方向纠正、宽高变化和 BGR→Bitmap 色彩/像素位置。
2. 实际打包的 YOLO/PP-OCRv4 模型处理 `demo_cabinet.jpg`，仍得到两个下方铭牌的几何匹配，两对文字识别通过，识别网络执行两次；JSON 含完整匹配和视频帧信息。
3. 开启实际 CameraX 预览，连续处理两帧；每帧人为增加 **2000 ms** 等待来模拟慢模型。推理进行时预览为 STREAMING；第二帧相机时间戳比第一帧前进超过 1 秒，排除紧邻旧帧排队的情况。暂停后等待 2.5 秒，显示结果不被旧会话更新；进入后台后相机解除绑定。

摄像头绑定成功与首帧预览显示之间存在正常启动间隔，测试先等待 STREAMING 再开始验证慢推理。

模拟器实际提供 1280×720 分析帧，方向纠正后为 720×1280，体现请求 1920×1080 时的设备尺寸回退。保存的一帧约耗时 3934 ms，**其中包含测试人为添加的 2000 ms**；这个数字不是手机性能数据。模拟摄像头画面为彩色测试图，没有真实旋钮，因此该流测试用于验证采集、调度和生命周期，不用于评估柜子识别准确率。

## 验证文件

- `validation/checks.json`：测试数量及源项目校验。
- `validation/camera-demo-matches.json`：实际模型的两对示例匹配。
- `validation/camera-live-result.json`：模拟摄像头第二帧的完整输出。
- `validation/camera-live-frame.png`：对应处理帧与标注。
- `validation/camera-screen.png`：实时预览、暂停状态、延迟指标与结果年龄的界面截图，已查看。

从相册先匹配版复制的 46 个源文件已逐一核对 SHA256，源项目文件没有变化。新工程的模型与匹配实现沿用源工程，仅包名调整；CPU 模式保持不变。

本次没有连接到物理手机，未报告手机实测速度、发热或真实柜子视频识别准确率。安装 arm64 APK 后应在手机上验证相机权限、对焦、运动模糊、实际帧尺寸和连续运行情况。
