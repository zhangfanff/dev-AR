package com.example.cabinetmatcher.camera.core;

import org.opencv.core.Mat;
import java.util.List;

/** A platform OCR adapter. Calls run on the pipeline's coordinator, never the UI thread. */
public interface TextEngine extends AutoCloseable {
    List<double[][]> detect(Mat bgr, int limitSide, TimingReport timing) throws Exception;
    TensorDecoders.Recognition recognize(Mat crop, TimingReport timing) throws Exception;
    @Override void close();
}
