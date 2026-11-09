# 实时预览叠加版验证

2026-10-19；versionCode 3，versionName `1.2-camera-live-overlay`。

- Debug ARM64/x86_64 APK、测试 APK 构建成功；33 项 JVM 单元测试全部通过。lint 为 0 错误、24 项警告。见 `validation/live-overlay-build.log`。
- 本地 Android 模拟器执行 4 项叠加专项测试：非零裁剪偏移配合四种旋转、FIT_CENTER 黑边的坐标换算、实际 Canvas 像素绘制及清除、导出 JSON 的叠加策略与处理帧坐标。全部通过，见 `validation/live-overlay-emulator-tests.log`。
- 本地模拟摄像头执行 1 项端到端测试：慢推理期间预览继续 STREAMING、连续两帧时间戳新鲜、结果同时发布到预览叠加、暂停/切换模型期间取消旧帧、加载新模型并更新结果、进入后台清空叠加并关闭摄像头。通过，见 `validation/live-overlay-camera-test.log`。这是合成场景测试，不代表开关柜真实识别准确率或真机速度。
- 模拟器冷启动期间出现过 System UI ANR 提示，保存的 `validation/live-overlay/camera-screen.png` 被系统弹窗遮挡。因此该截图不能用于评估无遮挡视觉效果或预览流畅度；上述自动断言通过也不等于真机视觉流畅度已确认。
- 已将 ARM64 正式调试 APK 更新安装到连接的 PHB110，包管理器确认版本 3。遵循用户此前选择，未向手机安装测试辅助 APK，未执行手机摄像头测试。实际柜子移动时的错位、延迟及帧率由用户在手机上观察。
- 模型文件和推理/匹配流程沿用 1.1 的四组合版本，未重新训练或更换权重。旧 1.1 APK 与源码 zip 已保留。

JSON 新字段与使用方式见 `LIVE_OVERLAY.md`。结果年龄是收到分析帧以来的时间，预览框代表最新完成的历史帧，不进行运动跟踪或补偿。下方 PNG 导出仍与原处理帧坐标对应。
