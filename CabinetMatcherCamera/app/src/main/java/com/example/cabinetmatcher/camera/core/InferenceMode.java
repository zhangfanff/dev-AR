package com.example.cabinetmatcher.camera.core;

/** Same-frame scheduling only. Frames themselves never run concurrently. */
public enum InferenceMode {
    SERIAL("串行（原流程对照）"), PARALLEL("并行（YOLO＋OCR＋手部）");
    public final String label;
    InferenceMode(String label) { this.label = label; }
}
