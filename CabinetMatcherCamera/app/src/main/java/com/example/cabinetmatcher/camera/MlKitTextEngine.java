package com.example.cabinetmatcher.camera;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.Rect;
import com.example.cabinetmatcher.camera.core.TensorDecoders;
import com.example.cabinetmatcher.camera.core.TextEngine;
import com.example.cabinetmatcher.camera.core.TimingReport;
import com.example.cabinetmatcher.camera.core.TextDetectionSizeMode;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/** Bundled Chinese SDK: no Play Services model download and no image upload. */
final class MlKitTextEngine implements TextEngine {
    private final TextRecognizer recognizer = TextRecognition.getClient(new ChineseTextRecognizerOptions.Builder().build());

    @Override public List<double[][]> detect(Mat bgr, int limitSide, TimingReport timing) throws Exception {
        long started = System.nanoTime();
        int[] size = TextDetectionSizeMode.inputSize(bgr.cols(), bgr.rows(), limitSide, false);
        int width = size[0], height = size[1];
        timing.detectionSourceWidth = bgr.cols(); timing.detectionSourceHeight = bgr.rows();
        timing.detectionInputWidth = width; timing.detectionInputHeight = height;
        Mat resized = new Mat();
        Bitmap bitmap = null;
        try {
            Imgproc.resize(bgr, resized, new Size(width, height));
            bitmap = bitmap(resized);
            timing.record("ocr_detection.preprocess", started);
            started = System.nanoTime();
            // The SDK also reads text here. Discard those strings: geometric matching uses only positions.
            Text output;
            try { output = process(bitmap); }
            finally { timing.record("ocr_detection.inference", started); }
            started = System.nanoTime();
            List<double[][]> boxes = new ArrayList<>();
            for (Text.TextBlock block : output.getTextBlocks()) for (Text.Line line : block.getLines()) {
                Point[] corners = line.getCornerPoints();
                if (corners == null || corners.length != 4) {
                    Rect rect = line.getBoundingBox();
                    if (rect == null) continue;
                    corners = new Point[]{new Point(rect.left, rect.top), new Point(rect.right, rect.top),
                            new Point(rect.right, rect.bottom), new Point(rect.left, rect.bottom)};
                }
                double[][] polygon = new double[4][2];
                for (int i = 0; i < 4; i++) {
                    polygon[i][0] = Math.max(0, Math.min(bgr.cols() - 1, corners[i].x * (double) bgr.cols() / width));
                    polygon[i][1] = Math.max(0, Math.min(bgr.rows() - 1, corners[i].y * (double) bgr.rows() / height));
                }
                if (Math.hypot(polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1]) > 3
                        && Math.hypot(polygon[3][0] - polygon[0][0], polygon[3][1] - polygon[0][1]) > 3) boxes.add(polygon);
            }
            boxes.sort(Comparator.comparingDouble((double[][] box) -> box[0][1]).thenComparingDouble(box -> box[0][0]));
            timing.record("ocr_detection.postprocess", started);
            return boxes;
        } finally { if (bitmap != null) bitmap.recycle(); resized.release(); }
    }

    @Override public TensorDecoders.Recognition recognize(Mat crop, TimingReport timing) throws Exception {
        long started = System.nanoTime();
        // Small OCR crops may be less than the SDK's minimum image size. Upscale and pad in white.
        double scale = Math.max(1, 48.0 / crop.rows());
        int width = Math.max(1, (int) Math.ceil(crop.cols() * scale));
        int height = Math.max(1, (int) Math.ceil(crop.rows() * scale));
        if (width > 4096) throw new IllegalArgumentException("文字区域过长，请缩小识别范围");
        Mat resized = new Mat(), padded = new Mat();
        Bitmap bitmap = null;
        try {
            Imgproc.resize(crop, resized, new Size(width, height), 0, 0, Imgproc.INTER_CUBIC);
            Core.copyMakeBorder(resized, padded, 8, 8, 8, Math.max(8, 32 - width - 8), Core.BORDER_CONSTANT, Scalar.all(255));
            bitmap = bitmap(padded);
            timing.record("recognition.preprocess", started);
            started = System.nanoTime();
            timing.recognitionModelCalls++;
            Text output;
            try { output = process(bitmap); }
            finally { timing.record("recognition.inference", started); }
            started = System.nanoTime();
            StringBuilder text = new StringBuilder();
            double weightedScore = 0;
            int characters = 0;
            for (Text.TextBlock block : output.getTextBlocks()) for (Text.Line line : block.getLines()) {
                String value = line.getText().trim();
                if (value.isEmpty()) continue;
                if (text.length() > 0) text.append('\n');
                text.append(value);
                int count = value.codePointCount(0, value.length());
                weightedScore += line.getConfidence() * count;
                characters += count;
            }
            timing.record("recognition.postprocess", started);
            return new TensorDecoders.Recognition(text.toString(), characters == 0 ? 0 : weightedScore / characters);
        } finally {
            if (bitmap != null) bitmap.recycle();
            resized.release(); padded.release();
        }
    }

    private static Bitmap bitmap(Mat bgr) {
        Mat rgba = new Mat();
        Bitmap bitmap = null;
        try {
            Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA);
            bitmap = Bitmap.createBitmap(bgr.cols(), bgr.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgba, bitmap);
            return bitmap;
        } catch (RuntimeException | OutOfMemoryError error) {
            if (bitmap != null) bitmap.recycle();
            throw error;
        } finally { rgba.release(); }
    }

    private Text process(Bitmap bitmap) throws Exception {
        Task<Text> task = recognizer.process(InputImage.fromBitmap(bitmap, 0));
        boolean interrupted = false;
        try {
            for (;;) {
                try {
                    Text result = Tasks.await(task);
                    if (interrupted) throw new CancellationException("OCR 等待已取消");
                    return result;
                } catch (InterruptedException error) {
                    // Drain the native call before the caller may recycle its input or close the client.
                    interrupted = true;
                } catch (ExecutionException error) {
                    Throwable cause = error.getCause();
                    if (cause instanceof Exception) throw (Exception) cause;
                    throw error;
                }
            }
        } finally { if (interrupted) Thread.currentThread().interrupt(); }
    }

    @Override public void close() { recognizer.close(); }
}
