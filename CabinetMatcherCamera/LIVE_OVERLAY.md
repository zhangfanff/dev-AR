# 实时预览叠加版（1.2-camera-live-overlay）

此版在 `CabinetMatcherCamera` 原项目更新。打开摄像头并开始识别后，上方实时画面上直接绘制旋钮框、已匹配文本框、匹配箭头以及 `K → R + 识别文字`。同一匹配使用相同颜色，未匹配旋钮或文字未通过识别使用橙色。下方仍保留该结果对应的原始处理帧与完整标注，可导出 JSON/PNG。YOLO 640/1280、OCR v4/v6 tiny 的模型选择继续保留。

预览区角落显示：识别/暂停状态、最近完成的帧编号、结果年龄、当前识别会话平均结果更新率、本帧处理时间、匹配数和所用模型。结果年龄每 0.1 秒刷新，检测框只在推理完成时更新；暂停时保持最后一组框并持续增加年龄。开始新会话、切换模型、切换分析分辨率或关闭摄像头时清除预览旧框。下方历史帧仍可核对和导出。

**移动相机时，旧框会暂时留在原位置，下一次推理完成后跳到新位置。这正是用于观察推理滞后的显示方式。** 没有加入跟踪或运动补偿。结果年龄从分析器收到该帧开始计时，包括本次处理和完成后等待，不包含曝光和相机驱动在帧送达前的时间。显示的更新率是结果更新 FPS，不能当作预览 FPS；实时取景继续由 CameraX 播放。

坐标通过 CameraX 共享 ViewPort 与 OutputTransform 映射，处理分析帧裁剪、0/90/180/270° 旋转以及预览 FIT_CENTER 的黑边。透明 View 只画矢量框和文字，不会用旧截图盖住实时画面，也不会截获点击对焦。模型和匹配流程不变；一次一帧，完成后处理最新帧。

JSON 的 `video.result_overlay_target` 更新为 `live_preview_and_processed_frame`，新增 `video.live_overlay` 描述叠加方式、无跟踪/运动补偿和年龄基准。所有检测/匹配坐标仍为处理帧像素坐标，不是手机屏幕坐标。PNG 仍是历史帧示意图，不是实时预览截图。

修改前的 1.1 APK 与完整源码压缩包保留在工作区 `outputs/CabinetMatcherCamera-model-switch-*`；此次交付使用 `CabinetMatcherCamera-live-overlay-*` 名称。

参考：[CameraX 坐标转换](https://developer.android.com/reference/androidx/camera/view/transform/CoordinateTransform)、[ImageProxyTransformFactory](https://developer.android.com/reference/androidx/camera/view/transform/ImageProxyTransformFactory)。实现使用项目已固定的 CameraX 1.4.2。
