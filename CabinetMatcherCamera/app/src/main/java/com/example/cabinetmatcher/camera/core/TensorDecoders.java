package com.example.cabinetmatcher.camera.core;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class TensorDecoders {
    private TensorDecoders() {}
    public static List<Detection> yolo(FloatBuffer values, int channels, int count, int width, int height,
                                        double scale, int padX, int padY, double threshold, double iouLimit) {
        if (channels != 5) throw new IllegalArgumentException("需要单类别 YOLO 输出 [1,5,N]");
        List<Detection> raw = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            double score = values.get(4 * count + i);
            if (score < threshold) continue;
            double cx = values.get(i), cy = values.get(count + i);
            double bw = values.get(2 * count + i), bh = values.get(3 * count + i);
            double[] box = {clip((cx - bw / 2 - padX) / scale, width), clip((cy - bh / 2 - padY) / scale, height),
                            clip((cx + bw / 2 - padX) / scale, width), clip((cy + bh / 2 - padY) / scale, height)};
            if (box[2] > box[0] && box[3] > box[1]) raw.add(new Detection(raw.size(), score, box, null, null));
        }
        raw.sort(Comparator.comparingDouble((Detection item) -> -item.confidence).thenComparingInt(item -> item.id));
        List<Detection> kept = new ArrayList<>();
        for (Detection candidate : raw) {
            boolean suppress = false;
            for (Detection previous : kept) if (iou(candidate.box, previous.box) > iouLimit) { suppress = true; break; }
            if (!suppress) kept.add(candidate);
            if (kept.size() == 300) break;
        }
        kept.sort(Comparator.comparingDouble((Detection item) -> item.cy).thenComparingDouble(item -> item.cx));
        for (int i = 0; i < kept.size(); i++) kept.get(i).id = i;
        return kept;
    }
    public static double iou(double[] a, double[] b) {
        double intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
                              Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        double union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
        return union > 0 ? intersection / union : 0;
    }
    public static Recognition ctc(FloatBuffer probabilities, int steps, int classes, List<String> dictionary) {
        if (classes != dictionary.size()) throw new IllegalArgumentException("识别字典与模型输出类别数不匹配");
        StringBuilder text = new StringBuilder();
        int previous = -1, accepted = 0;
        double totalScore = 0;
        for (int step = 0; step < steps; step++) {
            int best = 0;
            float score = probabilities.get(step * classes);
            for (int c = 1; c < classes; c++) {
                float value = probabilities.get(step * classes + c);
                if (value > score) { score = value; best = c; }
            }
            if (best != 0 && best != previous) { text.append(dictionary.get(best)); totalScore += score; accepted++; }
            previous = best;
        }
        return new Recognition(text.toString(), accepted == 0 ? 0 : totalScore / accepted);
    }
    public static double clip(double value, int size) { return Math.max(0, Math.min(size - 1, value)); }
    public static final class Recognition {
        public final String text;
        public final double confidence;
        public Recognition(String text, double confidence) { this.text = text; this.confidence = confidence; }
    }
}
