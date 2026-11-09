# 四方向文字检测验证

2026-11-09，版本 `1.9-four-direction-text`，versionCode 11。

## 已完成

- `testDebugUnitTest`、`assembleDebug`、`assembleDebugAndroidTest`、`lintDebug` 成功。76 项 JVM 测试，0 失败；lint 0 错误、24 警告。日志：`validation/four-direction-build.log`、`validation/four-direction-checks.log`。
- 新增 9 项 JVM 测试覆盖非正方形图的四向坐标反变换、像素边界、两种环绕次序的四边形 IoU、外接矩形重叠但实际不重叠的倾斜行、跨方向去重、原方向同次分割保留、默认设置和原方向的一对一旋钮匹配。
- Android 36.1 x86_64 无窗口模拟器（WHPX、4 核、4 GB）运行 `FourDirectionTextTest`，4 项全部通过，52.545 秒。日志：`validation/four-direction-android-tests.log`。
- 原生像素测试确认四个方向实际旋转、90°/270° 宽高交换、输入源不变；模拟只在 90° 检出的额外区域成功还原并保留，重复区域合并。单方向保持原列表及几何。异常中止后不继续检测或修改源图。
- 菜单测试确认缺省单方向、四方向选择在 Activity 重建及全新启动后保留，并可切回单方向。
- 真实图片测试使用 `demo_cabinet.jpg`、YOLO 640、OCR 检测长边 1600、默认方向纠正，分别运行 PP-OCRv4 和 ML Kit 的单方向串行、四方向串行、四方向并行。核对实际检测次数、合并计数、原图坐标范围、逐框识别无异常、每个选中框只识别一次、JSON 后端实际调用数；两后端四方向的串并行框及识别文字一致。
- 菜单截图 `validation/four-direction-menu.png`、`validation/four-direction-options.png` 已目视检查，两个选项完整显示。

## 样图结果

| OCR | 检测模式 | 各次检测累计框数 | 最终框数 | 几何匹配 | OCR 阈值通过 |
|---|---|---:|---:|---:|---:|
| PP-OCRv4 | 单方向 | 19 | 19 | 3 | 3 |
| PP-OCRv4 | 四方向（串并行一致） | 82 | 34 | 3 | 3 |
| ML Kit | 单方向 | 16 | 16 | 3 | 0 |
| ML Kit | 四方向（串并行一致） | 67 | 19 | 3 | 2 |

原始逐帧结果及摘要：`validation/four-direction-results.json`、`validation/four-direction-summary.json`。OCR 阈值通过只是应用置信度判定，不代表文字完全正确；ML Kit 四方向结果仍有错字。新增候选框也不能直接算作找回了同等数量的真实文字。

这次是功能和样图回归验证，没有标注旋转文字数据集或真实手机测试，不提供现场召回率/准确率提升比例，也不能将模拟器耗时当作手机 FPS。PP-OCRv5/v6、DBNet + CRNN 共用同一四方向调用层，本次没有对全部 OCR 组合重新运行真实模型测试。

## 安装包

手机 ARM64：`dist/CabinetMatcherCamera-four-direction-arm64-debug.apk`，355701630 字节。

SHA-256：`58fac78c6dfbcb018c0eb79971bf0ee946d1ac113c38faa29ac2918bb1f9b7f5`。另有 x86_64 和 universal APK 位于 `app/build/outputs/apk/debug/`。没有连接或安装到用户真实手机。

使用：暂停识别，在顶部 OCR 模型下方的“文字检测方向”选择单方向或四方向，再开始识别。详见 `FOUR_DIRECTION_TEXT.md`。
