from rfdetr import RFDETRSegPreview
import multiprocessing
import torch


# Windows系统多进程兼容配置（必须放在保护块前或内部）
def setup_multiprocessing():
    if torch.cuda.is_available():
        # GPU环境下启用spawn多进程（兼容Windows）
        torch.multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.freeze_support()  # 避免进程重复启动


if __name__ == "__main__":
    # 1. 初始化多进程配置（关键修复）
    # setup_multiprocessing()

    # 2. 初始化模型（你的原代码）
    model = RFDETRSegPreview()
    # model = RFDETRBase(pretrain_weights='rf-detr-base-2.pth')  # 可选权重
    # model = RFDETRLarge(pretrain_weights='rf-detr-large.pth')  # 可选大模型

    # 3. 启动训练（添加num_workers=0兜底，若仍报错可启用）
    model.train(
        dataset_dir="dataset_new",
        epochs=10,
        batch_size=1,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="model_ceshi",
        num_workers=0,  # 若仍报错，改为 num_workers=0（禁用多进程）
    )
