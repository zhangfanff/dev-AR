from rfdetr import RFDETRSegPreview

model = RFDETRSegPreview(pretrain_weights="model_new648/checkpoint_best_total.pth")

model.export()
