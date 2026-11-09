package com.example.cabinetmatcher.camera.core;

public enum TextDetectionMode {
    SINGLE("单方向检测（原方式）", new int[]{0}),
    FOUR_DIRECTIONS("四方向检测后合并（0° / 90° / 180° / 270°）", new int[]{0, 90, 180, 270});

    public final String label;
    private final int[] rotations;
    TextDetectionMode(String label, int[] rotations) { this.label = label; this.rotations = rotations; }
    public int[] rotations() { return rotations.clone(); }
    public static TextDetectionMode fromStoredId(String id) {
        for (TextDetectionMode mode : values()) if (mode.name().equals(id)) return mode;
        return SINGLE;
    }
}
