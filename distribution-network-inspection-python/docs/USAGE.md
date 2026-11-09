# 配网巡检模型与评测工具

## 环境

各模型使用独立环境，已用版本见 `environment_inventory.json`。PyTorch、Torchvision 与 CUDA 需配套安装。RF-DETR 使用支持 `RFDETRSegPreview` 的 1.3.0 版本。先运行 `python third_party/setup_dependencies.py` 恢复固定版本框架。第三方源码位于不提交的 `.external` 目录，代码范围见 `CODE_LAYOUT.md`。

## 数据路径

数据集和权重单独存放，不随代码提交。

- 检测数据：`images/train`、`images/val`、`images/test` 及对应的 `labels` 目录。
- 导线分割数据：COCO 标注及图像目录。
- 方向实验：`direction/dataset/manifest.json`、人工方向标注和预测掩码。

检测数据通过环境变量 `TOWER_DATASET_ROOT`、`TOWER_TOP_DATASET_ROOT` 指定。外部 YOLACT 和 RF-DETR 兼容数据加载器使用 `WIRE_DATASET_ROOT`。Mask R-CNN 的路径配置位于 `segmentation/maskrcnn/configs/wire700.json`。

其他路径可通过 `local_paths.example.json` 建立本机映射：

```shell
python tools/configure_paths.py --mapping local_paths.json --check
python tools/configure_paths.py --mapping local_paths.json --apply
```

## 目标检测

在 `detection/yolo` 目录运行：

```shell
python train_yolo12.py --task tower_top --weights /path/to/yolov12n.pt
python train_yolov8_baseline.py --task tower_top
python train_yolov5_baseline.py --task tower
python val_yolo12.py --task tower_top --weights /path/to/best.pt --data-root /path/to/dataset
```

任务参数集中在 `baseline_scripts/common.py`。YOLOv5 使用原版实现，初始化后位于 `.external/yolov5`。`--dry-run` 检查数据并输出配置，不启动训练。

## 导线分割

各模型在各自目录独立运行，避免 `data`、`utils` 等同名模块冲突。

- RF-DETR：`train_segmentation.py` 训练，`export_onnx.py` 导出；`train_ttpla_cable_matched.py` 与 `evaluate_local_data.py` 用于同设置的数据补充对照。
- Mask R-CNN：`train.py` 训练，`eval.py` 评测。配置文件为 `configs/wire700.json`。
- YOLACT：`train.py` 和 `eval.py`，数据配置为外部框架中的 `data/config.py`，可通过 `WIRE_DATASET_ROOT` 指定数据。

各入口可通过 `--help` 查看参数。`reference_scripts` 存放保留的早期入口，使用前应核对路径和参数。

## 方向估计

在 `direction` 目录运行标注和实验脚本：

- `annotate_angles.py`：方向标注。
- `check_annotations.py`：标注检查。
- `prepare_predictions.py`：生成预测掩码缓存。
- `run_experiments.py`：算法对比、区间统计和参数敏感性分析。

`wirelab/legacy.py` 为区域 PCA 与固定票箱投票；`wirelab/probabilistic.py` 为概率聚合。角度和评测定义见 `direction/docs/实验方法说明.md`。跨语言校验脚本需要外部 Kotlin 参考文件及 JVM 环境。

## 其他工具

- `deployment/onnx_tools`：ONNX 导出校验、分辨率评测、设备结果汇总和绘图。
- `evaluation/daytime_scenes`：人工场景分组及指标统计，输入为样本清单、标注和预测缓存。
- `tools/dataset/audit_duplicates.py`：按文件哈希检查训练集与验证集重复图像，只输出清单。

单元测试使用合成数据；需要外部数据的检查在输入缺失时跳过。模型精度和运行性能评测需提供对应数据、权重及设备结果。
