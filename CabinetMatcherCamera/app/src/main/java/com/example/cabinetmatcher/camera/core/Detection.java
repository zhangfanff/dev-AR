package com.example.cabinetmatcher.camera.core;

public final class Detection {
    public int id;
    public double confidence;
    public final double[] box;
    public final double[][] polygon;
    public String text;
    public final boolean textRegion;
    public boolean recognitionAttempted;
    public String recognitionStatus = "not_requested";
    public String recognitionError;
    public final double cx, cy;

    public Detection(int id, double confidence, double[] box, double[][] polygon, String text) {
        this.id = id;
        this.confidence = confidence;
        this.box = box;
        this.polygon = polygon;
        this.text = text;
        this.textRegion = polygon != null || text != null;
        this.recognitionAttempted = text != null;
        if (text != null) this.recognitionStatus = "recognized";
        double x = 0, y = 0;
        if (polygon != null) {
            for (double[] point : polygon) { x += point[0]; y += point[1]; }
            cx = x / polygon.length; cy = y / polygon.length;
        } else { cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2; }
    }
}
