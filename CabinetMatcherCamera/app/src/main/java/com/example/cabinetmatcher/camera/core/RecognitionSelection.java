package com.example.cabinetmatcher.camera.core;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** The geometric assignment is fixed before OCR; recognition never reassigns a pair. */
public final class RecognitionSelection {
    private RecognitionSelection() {}
    public static List<Detection> selected(MatchReport report) {
        List<Detection> selected = new ArrayList<>();
        Set<Integer> ids = new HashSet<>();
        for (MatchReport.Pair pair : report.matches) {
            if (!ids.add(pair.text.id)) throw new IllegalStateException("A region was assigned twice");
            selected.add(pair.text);
        }
        return selected;
    }
    public static void record(Detection region, String text, double confidence, double threshold) {
        region.recognitionAttempted = true;
        region.text = text == null ? "" : text;
        region.confidence = confidence;
        region.recognitionError = null;
        region.recognitionStatus = region.text.isEmpty() ? "empty_text" : confidence < threshold ? "low_confidence" : "recognized";
    }
    public static void failed(Detection region, Exception error) {
        region.recognitionAttempted = true; region.text = null; region.confidence = 0;
        region.recognitionStatus = "error";
        region.recognitionError = error.getClass().getSimpleName() + ": " + error.getMessage();
    }
}
