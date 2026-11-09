# YOLO 训练与验证

## 任务

`tower` 为单类塔身检测；`tower_top` 为塔顶组件检测，类别顺序为 `tower-top`、`hengdan`、`jinggai`、`car`、`byq`。

## 入口

```shell
python train_yolo12.py --task tower
python train_yolov8_baseline.py --task tower
python train_yolov5_baseline.py --task tower
python val_yolo12.py --task tower --weights /path/to/best.pt
python val_yolov8_baseline.py --task tower --split test --weights /path/to/best.pt
python val_yolov5_baseline.py --task tower --split test --weights /path/to/best.pt
```

塔顶任务将 `--task` 改为 `tower_top`。通过 `--epochs`、`--batch`、`--device`、`--workers` 设置运行参数。训练和评测参数集中在 `baseline_scripts/common.py`。

数据根目录由 `TOWER_DATASET_ROOT`、`TOWER_TOP_DATASET_ROOT` 指定。YOLOv5 使用独立进程运行原版代码，避免与 Ultralytics 模块冲突。

## 指标

AP 使用 COCOeval，预测置信度下限为 0.001，NMS IoU 为 0.7，最多保留 100 个预测。固定工作点的 P/R 使用置信度 0.25、匹配 IoU 0.50。塔顶主目标为类别 0，五类均值单独保存。

输出包含指标、预测和评测配置。使用验证集选择模型，再对测试集评测。已有输出目录时通过 `--name` 指定新名称。
