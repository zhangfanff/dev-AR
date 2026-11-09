# 代码范围

## 项目实现

- `detection/yolo/baseline_scripts`：检测任务配置、训练调度、原版 YOLOv5 兼容与统一评测。
- `segmentation/maskrcnn/wireseg`：导线数据加载、训练、COCO 指标和验证流程，模型结构调用 Torchvision。
- `segmentation/rfdetr`：分割训练入口、本地数据补充对照、评测与导出。
- `segmentation/yolact/coco_evaluation.py`：项目共享评测实现；其余入口调用外部 YOLACT。
- `direction/wirelab`：区域 PCA、固定票箱投票、概率聚合和评测。
- `direction`、`evaluation`、`deployment/onnx_tools`、`tools`：标注、分组实验、部署配套与数据工具。

## 第三方实现

YOLOv12、原版 YOLOv5 和 YOLACT 框架不直接放在仓库里。`third_party/dependencies.json` 固定来源、参考提交和文件校验值；`third_party/patches` 保存本地适配。

原目录没有 Git 历史，补丁是本地版本相对所选参考提交的差异，包含兼容处理、配置和版本差异，不等同于全部原创代码。YOLO、RF-DETR、YOLACT、Mask R-CNN 的模型架构归相应上游项目。

初始化：

```shell
python third_party/setup_dependencies.py
```

需要 Git 和网络，也可用 `--archive-dir` 指定预先下载的归档。框架恢复到 `.external`，该目录已被 Git 忽略，不需要上传。配置环境并准备数据、权重后，再运行各任务入口。

RF-DETR 使用 Python 包 `rfdetr==1.3.0`；Mask R-CNN 使用 Torchvision。原有模型版本与数据划分保持不变。许可证见各模块 LICENSE 及 `third_party/NOTICES.txt`。
