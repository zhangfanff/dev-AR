# 跟随相机分析帧尺寸验证

2026-11-09，版本 `1.10-frame-text-size`，versionCode 12。

## 变更

新增可保存的“文字检测尺寸”下拉框，保留手动长边上限默认行为，可选择跟随实际分析帧。跟随时不应用手动 1600/2048 限制，PP-OCR / DBNet 继续按原规则对齐 32，ML Kit 按帧尺寸提交 SDK。单方向和四方向均适用。详细计时及 JSON 逐次记录源图与检测输入宽高。

## 已验证

- `testDebugUnitTest`、`assembleDebug`、`assembleDebugAndroidTest`、`lintDebug` 成功。83 项 JVM 测试，0 失败；lint 0 错误、24 警告。见 `validation/frame-text-size-build.log`、`validation/frame-text-size-checks.log`。
- 新增 7 项 JVM 测试，验证原手动规则、1920×1080 跟随、3840×2160 不受 2048 上限限制的尺寸计算、横竖交换、小帧不主动放大、旧设置默认恢复、跟随时忽略无效的未启用手动限制。
- Android 36.1 x86_64 无窗口模拟器，WHPX，4 核 / 4 GB，运行 `FourDirectionTextTest` 和 `FrameTextSizeTest`，6 项全部通过，52.296 秒。覆盖原四方向回归、真实模型跟随尺寸、尺寸选项保存和手动输入框启停。见 `validation/frame-text-size-android-tests.log`。
- 菜单实测默认手动模式；选择跟随后手动输入变灰；Activity 重建及重新启动恢复所选模式，切回手动后输入恢复可用。
- 真实模型使用 `demo_cabinet.jpg`（1706×1279）、YOLO 640，设置未启用的手动上限为 640，确认跟随模式仍使用原帧尺寸。PP-OCRv4 和 ML Kit 分别运行单方向与四方向，均得到 3 对几何匹配，逐区域识别无异常。四方向 90°/270° 输入按实际源图交换宽高。

| 后端 | 源图 | 0°/180° 实际输入 | 90°/270° 实际输入 |
|---|---|---|---|
| PP-OCRv4 | 1706×1279 | 1696×1280 | 1280×1696 |
| ML Kit SDK | 1706×1279 | 1706×1279 | 1279×1706 |

原始结果：`validation/frame-text-size-results.json`；摘要：`validation/frame-text-size-summary.json`。原四方向测试还核对手动模式下串并行结果一致。

界面截图 `validation/frame-size-settings.png`、`validation/frame-size-options.png`、`validation/frame-size-follow.png` 已目视检查，下拉选项完整显示，跟随模式手动长边输入正确变灰。

这是模拟器功能验证，不是手机准确率或性能评测。没有连接真实手机；3840×2160 本次仅验证尺寸计算，未在真实模型上测试 4K 推理。PP-OCRv5/v6 和 DBNet + CRNN 共用输入尺寸逻辑，本次未逐一重新运行其真实模型。采集分辨率仍受设备和 CameraX 裁剪影响，本选项不新增采集档位。

## 安装包

`dist/CabinetMatcherCamera-frame-text-size-arm64-debug.apk`，355767631 字节。

SHA-256：`926f856c855aed2502eaba807c100a1976d6f11ce711cc13da21753d3bbbf2ff`。x86_64 和 universal APK 位于 `app/build/outputs/apk/debug/`。

操作：暂停识别，滚动到下方设置，“文字检测尺寸”选择“跟随相机分析帧（原尺寸）”，再开始识别。使用说明见 [FRAME_TEXT_SIZE.md](FRAME_TEXT_SIZE.md)。
