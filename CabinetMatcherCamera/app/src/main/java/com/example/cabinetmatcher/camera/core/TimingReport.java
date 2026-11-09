package com.example.cabinetmatcher.camera.core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Monotonic wall-clock durations in nanoseconds. Totals overlap their child stages. */
public final class TimingReport {
    public final Map<String, Long> stages = new LinkedHashMap<>();
    public final List<TextCost> textRegions = new ArrayList<>();
    public boolean modelsReused, angleClassifierEnabled;
    public int detectedTextRegionCount;
    public int recognitionModelCalls;
    public TextDetectionMode textDetectionMode = TextDetectionMode.SINGLE;
    public TextDetectionSizeMode textDetectionSizeMode = TextDetectionSizeMode.MANUAL_LIMIT;
    public int detectionSourceWidth, detectionSourceHeight, detectionInputWidth, detectionInputHeight;
    public int rawTextRegionCount;
    public final List<DetectionPass> detectionPasses = new ArrayList<>();

    public static final class DetectionPass {
        public final int rotationDegrees, regionCount;
        public final TimingReport timing;
        public DetectionPass(int rotationDegrees, int regionCount, TimingReport timing) {
            this.rotationDegrees = rotationDegrees; this.regionCount = regionCount; this.timing = timing;
        }
    }

    public void record(String key, long startedNs) { add(key, System.nanoTime() - startedNs); }
    public void add(String key, long durationNs) {
        if (durationNs < 0) throw new IllegalArgumentException("Negative duration");
        stages.put(key, nanos(key) + durationNs);
    }
    public long nanos(String key) { return stages.getOrDefault(key, 0L); }
    public double millis(String key) { return nanos(key) / 1_000_000.0; }
    public void copyFrom(TimingReport other) { stages.putAll(other.stages); }
    public void other(String key, String total, String... children) {
        long remaining = nanos(total);
        for (String child : children) remaining -= nanos(child);
        stages.put(key, Math.max(0L, remaining));
    }
    public static final class TextCost {
        public final int regionIndex;
        public final TimingReport timing = new TimingReport();
        public Integer textId;
        public String text = "";
        public double confidence;
        public boolean accepted;
        public String recognitionStatus, error;
        public int cropWidth, cropHeight;
        public TextCost(int regionIndex) { this.regionIndex = regionIndex; }
    }
}
