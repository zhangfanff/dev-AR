package com.example.cabinetmatcher.camera;

import com.example.cabinetmatcher.camera.core.Detection;
import java.util.Locale;

final class RecognitionLabel {
    static String shortStatus(Detection region) {
        switch (region.recognitionStatus) {
            case "recognized": return String.format(Locale.CHINA,"OCR通过 %.3f",region.confidence);
            case "low_confidence": return String.format(Locale.CHINA,"OCR低置信度 %.3f",region.confidence);
            case "empty_text": return "OCR未读出";
            case "error": return "OCR出错";
            default: return "OCR未执行";
        }
    }
    static String describe(Detection region) {
        switch (region.recognitionStatus) {
            case "recognized": return String.format(Locale.CHINA, "识别通过 · OCR %.3f", region.confidence);
            case "low_confidence": return String.format(Locale.CHINA, "识别置信度不足 · OCR %.3f（保留几何配对）", region.confidence);
            case "empty_text": return "未读出文字（保留几何配对）";
            case "error": return "识别出错（保留几何配对）：" + region.recognitionError;
            default: return "未执行文字识别";
        }
    }
}
