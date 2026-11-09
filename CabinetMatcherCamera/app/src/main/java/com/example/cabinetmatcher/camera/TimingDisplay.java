package com.example.cabinetmatcher.camera;

import com.example.cabinetmatcher.camera.core.TimingReport;
import java.util.Locale;

final class TimingDisplay {
    private static void row(StringBuilder out, TimingReport t, String label, String key) {
        out.append(String.format(Locale.CHINA, "%s：%.3f ms\n", label, t.millis(key)));
    }
    static String summary(TimingReport t) {
        StringBuilder out = new StringBuilder();
        out.append("整图区域 " + t.detectedTextRegionCount + " 个 · 识别网络 " + t.recognitionModelCalls + " 次 · 跳过 " + (t.detectedTextRegionCount - t.textRegions.size()) + " 个\n");
        out.append(t.modelsReused ? "模型已复用，本次加载为 0\n" : "本次包含首次模型准备与加载\n");
        row(out, t, "选图时读取与整理", "image_load.total");
        row(out, t, "模型文件准备/校验", "model_prepare");
        row(out, t, "YOLO/OCR 模型加载", "model_load.total");
        row(out, t, "图片格式转换", "image_convert");
        row(out, t, "旋钮检测", "yolo.total");
        row(out, t, "文字区域检测", "ocr_detection.total");
        row(out, t, "区域坐标整理", "region_prepare");
        row(out, t, "旋钮与文本匹配", "matching");
        row(out, t, "已选区域识别（" + t.textRegions.size() + " 个）", "text_loop.total");
        row(out, t, "标注图片绘制", "annotation");
        row(out, t, "结果控件更新", "ui_update");
        row(out, t, "等待与其他开销", "analysis.other");
        row(out, t, "检测＋OCR＋匹配合计", "pipeline.total");
        row(out, t, "本次处理合计（含加载/等待）", "analysis.total");
        out.append("选图读取单独计时，不计入本次处理合计。\n总计已含子步骤，不能重复相加；不含导出、相册操作和屏幕实际绘帧。");
        return out.toString();
    }
    static String details(TimingReport t) {
        StringBuilder out = new StringBuilder();
        String[][] rows = {
            {"图片尺寸读取", "image_load.bounds"}, {"EXIF 读取", "image_load.exif"},
            {"图片解码", "image_load.decode"}, {"方向纠正", "image_load.orientation"}, {"大图缩小", "image_load.resize"},
            {"字典加载", "model_load.dictionary"}, {"YOLO 加载", "model_load.yolo"},
            {"OCR 检测模型加载", "model_load.ocr_detection"}, {"OCR 识别模型加载", "model_load.ocr_recognition"},
            {"平台 OCR 客户端创建", "model_load.platform_ocr"},
            {"YOLO 预处理", "yolo.preprocess"}, {"YOLO 模型推理", "yolo.inference"}, {"YOLO 后处理/NMS", "yolo.postprocess"},
            {"YOLO 清理与其他", "yolo.other"},
            {"文字检测预处理", "ocr_detection.preprocess"}, {"文字检测模型推理", "ocr_detection.inference"},
            {"文字检测后处理", "ocr_detection.postprocess"}, {"文字检测清理与其他", "ocr_detection.other"},
            {"检测输入旋转", "ocr_detection.rotation"}, {"检测框还原原图坐标", "ocr_detection.mapping"},
            {"跨方向重复框合并", "ocr_detection.merge"},
            {"YOLO＋文字检测实际经过时间（含汇合）", "detection.wall"},
            {"手部独立输入位图转换", "hands.bitmap"},
            {"文字裁剪累计", "text_loop.crop"}, {"方向模型首次加载", "text_loop.angle.model_load"},
            {"方向预处理累计", "text_loop.angle.preprocess"}, {"方向推理累计", "text_loop.angle.inference"},
            {"方向后处理累计", "text_loop.angle.postprocess"},
            {"识别预处理累计", "text_loop.recognition.preprocess"}, {"识别模型推理累计", "text_loop.recognition.inference"},
            {"结果整理 / CTC 解码累计", "text_loop.recognition.postprocess"}, {"文本过滤累计", "text_loop.filter"},
            {"文字循环清理与其他", "text_loop.other"}, {"流程衔接与其他", "pipeline.other"}
        };
        for (String[] entry : rows) row(out, t, entry[0], entry[1]);
        out.append("\n文字检测：").append(t.textDetectionMode.label).append('\n');
        out.append("检测尺寸：").append(t.textDetectionSizeMode.label).append('\n');
        out.append("各方向累计 ").append(t.rawTextRegionCount).append(" 个框 → 最终 ")
                .append(t.detectedTextRegionCount).append(" 个区域\n");
        for (TimingReport.DetectionPass pass : t.detectionPasses) {
            if (pass.timing.detectionInputWidth > 0) out.append(String.format(Locale.CHINA,
                    "  %d° 尺寸：%d×%d → %d×%d\n",pass.rotationDegrees,
                    pass.timing.detectionSourceWidth,pass.timing.detectionSourceHeight,
                    pass.timing.detectionInputWidth,pass.timing.detectionInputHeight));
            out.append(String.format(Locale.CHINA, "  %d°：%d 个框 · %.3f ms\n",
                    pass.rotationDegrees, pass.regionCount, pass.timing.millis("total")));
        }
        out.append(t.angleClassifierEnabled ? "\n方向纠正已开启\n" : "\n方向纠正关闭，相关耗时为 0\n");
        out.append("\n已选文字区域（未选中的区域不运行识别网络）\n");
        for (TimingReport.TextCost cost : t.textRegions) {
            out.append(String.format(Locale.CHINA, "\n区域 %d%s · %.3f ms\n%s\n", cost.regionIndex,
                    " → R" + cost.textId + (cost.accepted ? "（识别通过）" : "（识别未通过）"), cost.timing.millis("total"), cost.text));
            row(out, cost.timing, "  裁剪", "crop");
            if (t.angleClassifierEnabled) {
                row(out, cost.timing, "  方向模型加载", "angle.model_load");
                row(out, cost.timing, "  方向预处理", "angle.preprocess");
                row(out, cost.timing, "  方向推理", "angle.inference");
                row(out, cost.timing, "  方向后处理", "angle.postprocess");
            }
            row(out, cost.timing, "  识别预处理", "recognition.preprocess");
            row(out, cost.timing, "  识别模型推理", "recognition.inference");
            row(out, cost.timing, "  结果整理 / CTC 解码", "recognition.postprocess");
            row(out, cost.timing, "  文本过滤", "filter");
            row(out, cost.timing, "  清理与其他", "other");
        }
        return out.toString();
    }
}
