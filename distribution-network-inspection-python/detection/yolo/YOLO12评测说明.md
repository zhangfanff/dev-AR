# YOLOv12 评测

```shell
python val_yolo12.py --task tower --weights /path/to/best.pt
python val_yolo12.py --task tower_top --weights /path/to/best.pt --data-root /path/to/dataset
```

默认使用测试集；验证集通过 `--split val` 指定。结果保存在 `runs/yolo12_eval`，包括指标、预测和生效配置。

同一批预测分别计算 COCO 和 Ultralytics 指标。两者的 AP 插值积分与匹配实现存在差异；P/R 也采用不同工作点：COCO 结果固定置信度 0.25、匹配 IoU 0.50，Ultralytics 使用最大平均 F1 对应的置信度。

`primary_target_metrics` 为类别 0 的 COCO 指标。塔顶任务类别 0 为柱体，不能与五类平均 AP 混用。准确率输出为百分数。

默认设置为输入 640、batch 1、FP32、NMS IoU 0.7、最多 100 个预测。`iou` 参数是 NMS 阈值；AP 仍按 0.50 至 0.95 的十个匹配 IoU 阈值计算。比较其他脚本结果时，应同时核对权重、数据划分、类别范围和工作点。
