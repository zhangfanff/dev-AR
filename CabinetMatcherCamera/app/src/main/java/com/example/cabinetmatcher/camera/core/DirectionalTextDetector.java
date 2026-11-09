package com.example.cabinetmatcher.camera.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import org.opencv.core.Core;
import org.opencv.core.Mat;

/** Reuses the selected detector sequentially; YOLO and recognition remain outside this loop. */
public final class DirectionalTextDetector {
    private DirectionalTextDetector() {}
    public interface Detector { List<double[][]> detect(Mat image, TimingReport timing) throws Exception; }

    public static List<double[][]> detect(Mat image, TextDetectionMode mode, Detector detector,
                                         TimingReport timing, CabinetPipeline.Progress progress) throws Exception {
        timing.textDetectionMode = mode;
        if (mode == TextDetectionMode.SINGLE) {
            long started = System.nanoTime();
            TimingReport pass = new TimingReport();
            List<double[][]> regions = detector.detect(image, pass);
            for (Map.Entry<String, Long> entry : pass.stages.entrySet()) timing.add(entry.getKey(), entry.getValue());
            pass.record("total", started);
            timing.detectionPasses.add(new TimingReport.DetectionPass(0, regions.size(), pass));
            timing.rawTextRegionCount = regions.size();
            return regions; // Preserve the original detector's ordering, geometry and segmentation.
        }
        List<TextRegionMerger.Region> candidates = new ArrayList<>();
        Mat rotated = new Mat();
        try {
            for (int degrees : mode.rotations()) {
                if (Thread.currentThread().isInterrupted()) throw new CancellationException("文字检测已取消");
                progress.update("正在检测文字区域：" + degrees + "°（四方向合并）…");
                TimingReport pass = new TimingReport();
                long passStarted = System.nanoTime(), started = passStarted;
                Mat input = image;
                if (degrees != 0) {
                    int code = degrees == 90 ? Core.ROTATE_90_CLOCKWISE
                            : degrees == 180 ? Core.ROTATE_180 : Core.ROTATE_90_COUNTERCLOCKWISE;
                    Core.rotate(image, rotated, code);
                    input = rotated;
                }
                pass.record("ocr_detection.rotation", started);
                List<double[][]> regions = detector.detect(input, pass);
                started = System.nanoTime();
                for (double[][] polygon : regions) candidates.add(new TextRegionMerger.Region(
                        TextRegionMerger.restore(polygon, degrees, image.cols(), image.rows()), degrees));
                pass.record("ocr_detection.mapping", started);
                for (Map.Entry<String, Long> entry : pass.stages.entrySet()) timing.add(entry.getKey(), entry.getValue());
                pass.record("total", passStarted);
                timing.detectionPasses.add(new TimingReport.DetectionPass(degrees, regions.size(), pass));
                timing.rawTextRegionCount += regions.size();
            }
        } finally { rotated.release(); }
        long started = System.nanoTime();
        List<double[][]> merged = TextRegionMerger.merge(candidates);
        timing.record("ocr_detection.merge", started);
        return merged;
    }
}
