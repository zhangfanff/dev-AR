# TTPLA 与混合数据对照

`train_ttpla_cable_matched.py` 从参考 checkpoint 读取模型及训练参数，使用 TTPLA 导线样本训练。验证集保持一致，测试集不参与选模。

```shell
python train_ttpla_cable_matched.py --help
python train_ttpla_cable_matched.py --check-model
python evaluate_local_data.py --help
```

需准备参考 checkpoint、通用预训练权重及数据。默认参考目录为 `model_new648`，TTPLA 对照输出为 `model_ttpla_cable_matched648`。输入尺寸 648、单类 cable、batch 1、梯度累积 4、10 个 epoch；具体生效参数记录在训练输出的协议文件中。

评测时将验证集选择的两份 checkpoint 应用于同一组本地测试图像，输出 COCO 指标及数据、权重摘要。
