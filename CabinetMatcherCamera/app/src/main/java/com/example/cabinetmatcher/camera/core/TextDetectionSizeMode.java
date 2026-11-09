package com.example.cabinetmatcher.camera.core;

/** Controls detector preprocessing, independently of model identity and detection rotations. */
public enum TextDetectionSizeMode {
    MANUAL_LIMIT("手动长边上限（原方式）"),
    FOLLOW_FRAME("跟随相机分析帧（原尺寸）");

    public final String label;
    TextDetectionSizeMode(String label) { this.label = label; }
    public int longSideLimit(int frameWidth, int frameHeight, int configuredLimit) {
        return this == FOLLOW_FRAME ? Math.max(frameWidth, frameHeight) : configuredLimit;
    }
    public static TextDetectionSizeMode fromStoredId(String id) {
        for (TextDetectionSizeMode mode : values()) if (mode.name().equals(id)) return mode;
        return MANUAL_LIMIT;
    }

    /** Same resize rules as the original detectors. Paddle needs multiples of 32; ML Kit does not. */
    public static int[] inputSize(int frameWidth, int frameHeight, int longSideLimit, boolean alignTo32) {
        double scale = Math.min(1.0, (double) longSideLimit / Math.max(frameWidth, frameHeight));
        if (alignTo32) return new int[]{
                Math.max(32, (int) Math.rint((int) (frameWidth * scale) / 32.0) * 32),
                Math.max(32, (int) Math.rint((int) (frameHeight * scale) / 32.0) * 32)};
        return new int[]{Math.max(32, (int) Math.round(frameWidth * scale)),
                Math.max(32, (int) Math.round(frameHeight * scale))};
    }
}
