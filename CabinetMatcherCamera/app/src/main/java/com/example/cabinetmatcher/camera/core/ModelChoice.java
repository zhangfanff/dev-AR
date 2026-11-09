package com.example.cabinetmatcher.camera.core;

/** Immutable model identity, also captured with each displayed/exported frame. */
public final class ModelChoice {
    public enum Ocr {
        V4("PP-OCRv4", "ch_PP-OCRv4_det_infer.onnx", "ch_PP-OCRv4_rec_infer.onnx", "rec_character_dict.txt", 6625,
                "原版模型，作为识别效果和耗时的对照。"),
        V6_TINY("PP-OCRv6 tiny", "PP-OCRv6_det_tiny.onnx", "PP-OCRv6_rec_tiny.onnx", "rec_character_dict_v6_tiny.txt", 6906,
                "轻量模型，适合优先比较实时速度。"),
        V5_MOBILE("PP-OCRv5 mobile", "ch_PP-OCRv5_det_mobile.onnx", "ch_PP-OCRv5_rec_mobile.onnx", "rec_character_dict_v5.txt", 18385,
                "新增移动端模型，可与 v4 / v6 tiny 对比中文铭牌效果。"),
        V5_SERVER("PP-OCRv5 server", "ch_PP-OCRv5_det_server.onnx", "ch_PP-OCRv5_rec_server.onnx", "rec_character_dict_v5.txt", 18385,
                "较大模型，检测和识别均使用 server 版；手机 CPU 耗时和内存占用较高。server 指模型规格，仍在本机离线运行。"),
        V6_SMALL("PP-OCRv6 small", "PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx", "rec_character_dict_v6_small.txt", 18710,
                "比 tiny 更大的 v6 模型，比较识别效果与耗时的取舍。所有模型均离线运行，效果请以现场测试为准。"),
        ML_KIT_CHINESE("Google ML Kit v2（中文）", null, null, null, 0,
                "内置中文模型，支持中文及拉丁字符，离线运行。SDK 整图调用同时检测和识别；用文字行位置匹配后，再识别选中区域。OCR 分数来自 ML Kit，与 Paddle 分数不可直接比较。"),
        DBNET_CRNN("DBNet + CRNN（中文）", "ch_PP-OCRv4_det_infer.onnx", "ch_ppocr_mobile_v2.0_rec_infer.onnx", "rec_character_dict_crnn.txt", 6625,
                "DB 系列检测（复用 v4 检测器）＋ MobileNetV3 / BiLSTM / CTC 中文 CRNN，识别输入高度 32。作为经典算法对照，不保证比新版 Paddle 更准。");
        public final String label, detectionFile, recognitionFile, dictionaryFile, description;
        public final int classes;
        Ocr(String label, String detectionFile, String recognitionFile, String dictionaryFile, int classes, String description) {
            this.label = label; this.detectionFile = detectionFile; this.recognitionFile = recognitionFile;
            this.dictionaryFile = dictionaryFile; this.classes = classes; this.description = description;
        }
        public boolean usesMlKit() { return this == ML_KIT_CHINESE; }
        public int recognitionHeight() { return this == DBNET_CRNN ? 32 : 48; }
        public String backend() { return usesMlKit() ? "ML Kit bundled Chinese" : "ONNX Runtime CPU"; }
        /** Stable model IDs survive list growth; retain the old boolean preference on upgrade. */
        public static Ocr fromStoredId(String id, boolean legacyV6) {
            if (id != null) try { return valueOf(id); } catch (IllegalArgumentException ignored) { }
            return legacyV6 ? V6_TINY : V4;
        }
    }
    public final int yoloSide;
    public final Ocr ocr;
    public ModelChoice(int yoloSide, Ocr ocr) {
        if (yoloSide != 640 && yoloSide != 1280) throw new IllegalArgumentException("YOLO 输入须为 640 或 1280");
        if (ocr == null) throw new IllegalArgumentException("请选择 OCR 模型");
        this.yoloSide = yoloSide; this.ocr = ocr;
    }
    public static ModelChoice defaults() { return new ModelChoice(1280, Ocr.V4); }
    public String yoloFile() { return "knob_" + yoloSide + ".onnx"; }
    public String label() { return "YOLOv8n " + yoloSide + " · " + ocr.label + (ocr.usesMlKit() ? " · 本机" : " · CPU"); }
    @Override public boolean equals(Object other) {
        if (!(other instanceof ModelChoice)) return false;
        ModelChoice value = (ModelChoice) other;
        return yoloSide == value.yoloSide && ocr == value.ocr;
    }
    @Override public int hashCode() { return yoloSide * 31 + ocr.ordinal(); }
}
