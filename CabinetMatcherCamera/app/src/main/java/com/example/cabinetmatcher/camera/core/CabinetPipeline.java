package com.example.cabinetmatcher.camera.core;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/** ONNX inference + OpenCV preprocessing. No Android dependency: also testable on desktop. */
public final class CabinetPipeline implements AutoCloseable {
    private final OrtEnvironment environment = OrtEnvironment.getEnvironment();
    private OrtSession yolo, detector, recognizer, classifier;
    private final File modelDir;
    private final TextEngine textEngine;
    public final ModelChoice models;
    private final List<String> dictionary = new ArrayList<>();
    public final TimingReport initializationTiming = new TimingReport();
    public interface Progress { void update(String message); }
    public static final class Settings {
        public ModelChoice models = ModelChoice.defaults();
        public double yoloConfidence = 0.25, ocrConfidence = 0.5, boxConfidence = 0.5;
        public double maxAngleFromDownwardDegrees = 45;
        public SearchDirection searchDirection = SearchDirection.DOWN;
        public TextDetectionMode textDetectionMode = TextDetectionMode.SINGLE;
        public TextDetectionSizeMode textDetectionSizeMode = TextDetectionSizeMode.MANUAL_LIMIT;
        public int detSide = 1600;
        public boolean angleClassifier = true;
        public InferenceMode inferenceMode = InferenceMode.SERIAL;
        public void validate() {
            if (inferenceMode == null) throw new IllegalArgumentException("请选择执行模式");
            if (models == null) throw new IllegalArgumentException("请选择模型组合");
            if (searchDirection == null) throw new IllegalArgumentException("请选择匹配方向");
            if (textDetectionMode == null) throw new IllegalArgumentException("请选择文字检测方向");
            if (textDetectionSizeMode == null) throw new IllegalArgumentException("请选择文字检测尺寸模式");
            for (double threshold : new double[]{yoloConfidence, ocrConfidence, boxConfidence})
                if (!Double.isFinite(threshold) || threshold < 0 || threshold > 1) throw new IllegalArgumentException("置信度须在 0 到 1 之间");
            if (textDetectionSizeMode == TextDetectionSizeMode.MANUAL_LIMIT && (detSide < 32 || detSide > 2048))
                throw new IllegalArgumentException("OCR 手动检测长边须在 32 到 2048 之间");
            if (!Double.isFinite(maxAngleFromDownwardDegrees) || maxAngleFromDownwardDegrees < 0 || maxAngleFromDownwardDegrees > 60)
                throw new IllegalArgumentException("搜索扇区半角须为 0 到 60 度");
        }
    }
    public CabinetPipeline(File modelDir) throws Exception {
        this(modelDir, ModelChoice.defaults());
    }
    public CabinetPipeline(File modelDir, ModelChoice models) throws Exception {
        this(modelDir, models, null);
    }
    public CabinetPipeline(File modelDir, ModelChoice models, TextEngine textEngine) throws Exception {
        this.modelDir = modelDir;
        this.models = java.util.Objects.requireNonNull(models);
        this.textEngine = textEngine;
        if (models.ocr.usesMlKit() != (textEngine != null)) {
            if (textEngine != null) textEngine.close();
            throw new IllegalArgumentException("ML Kit 模型须通过 Android OCR 工厂加载");
        }
        long loadStart = System.nanoTime();
        long started = System.nanoTime();
        try {
            if (textEngine == null) {
                dictionary.add(""); // CTC blank, index 0.
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                        new FileInputStream(new File(modelDir, models.ocr.dictionaryFile)), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) dictionary.add(line);
                }
                dictionary.add(" ");
                if (dictionary.size() != models.ocr.classes) throw new IllegalStateException("OCR 字典与模型不匹配：" + dictionary.size());
            }
            initializationTiming.record("model_load.dictionary", started);
            started = System.nanoTime();
            yolo = session(models.yoloFile());
            long[] yoloShape = ((TensorInfo) yolo.getInputInfo().values().iterator().next().getInfo()).getShape();
            if (!java.util.Arrays.equals(yoloShape, new long[]{1, 3, models.yoloSide, models.yoloSide}))
                throw new IllegalStateException("YOLO 文件输入尺寸不匹配");
            initializationTiming.record("model_load.yolo", started);
            if (textEngine == null) {
                started = System.nanoTime();
                detector = session(models.ocr.detectionFile);
                initializationTiming.record("model_load.ocr_detection", started);
                started = System.nanoTime();
                recognizer = session(models.ocr.recognitionFile);
                long[] recognitionShape = ((TensorInfo) recognizer.getOutputInfo().values().iterator().next().getInfo()).getShape();
                if (recognitionShape[recognitionShape.length - 1] != dictionary.size())
                    throw new IllegalStateException("OCR 输出类别数与字典不匹配");
                initializationTiming.record("model_load.ocr_recognition", started);
            } else {
                initializationTiming.add("model_load.ocr_detection", 0);
                initializationTiming.add("model_load.ocr_recognition", 0);
            }
            initializationTiming.record("model_load.total", loadStart);
        } catch (Exception | OutOfMemoryError error) {
            try { close(); } catch (Exception cleanup) { error.addSuppressed(cleanup); }
            throw error;
        }
    }
    private OrtSession session(String filename) throws OrtException {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(2);
            options.setInterOpNumThreads(1);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            return environment.createSession(new File(modelDir, filename).getAbsolutePath(), options);
        }
    }
    private OrtSession.Result runMeasured(OrtSession session, OnnxTensor tensor, TimingReport timing, String stage) throws OrtException {
        long started = System.nanoTime();
        if ("recognition".equals(stage)) timing.recognitionModelCalls++;
        try {
            return session.run(Collections.singletonMap(session.getInputNames().iterator().next(), tensor));
        } finally { timing.record(stage + ".inference", started); }
    }
    public MatchReport run(Mat bgr, Settings settings, Progress progress) throws Exception {
        return run(bgr, settings, progress, null);
    }
    /** OCR stays on the caller; only independent YOLO work goes to its worker. */
    public MatchReport run(Mat bgr, Settings settings, Progress progress, ExecutorService yoloWorker) throws Exception {
        settings.validate();
        if (settings.inferenceMode == InferenceMode.PARALLEL && yoloWorker == null)
            throw new IllegalArgumentException("并行模式需要独立的 YOLO 工作线程");
        if (!models.equals(settings.models)) throw new IllegalArgumentException("当前模型与所选设置不同，请等待模型加载完成");
        if (bgr.empty() || bgr.type() != CvType.CV_8UC3) throw new IllegalArgumentException("需要 BGR 三通道图片");
        long start = System.nanoTime();
        TimingReport timing = new TimingReport();
        timing.angleClassifierEnabled = settings.angleClassifier;
        timing.textDetectionSizeMode = settings.textDetectionSizeMode;
        for (String key : new String[]{"crop", "angle.model_load", "angle.preprocess", "angle.inference", "angle.postprocess",
                "recognition.preprocess", "recognition.inference", "recognition.postprocess", "filter"}) timing.add("text_loop." + key, 0);

        long detectionStarted = System.nanoTime();
        TimingReport yoloTiming = new TimingReport();
        Future<List<Detection>> yoloTask = null;
        if (settings.inferenceMode == InferenceMode.PARALLEL) {
            progress.update("正在并行检测旋钮与整图文字区域…");
            yoloTask = yoloWorker.submit(() -> {
                long started = System.nanoTime();
                List<Detection> found = detectKnobs(bgr, settings.yoloConfidence, yoloTiming);
                yoloTiming.record("yolo.total", started);
                yoloTiming.other("yolo.other", "yolo.total", "yolo.preprocess", "yolo.inference", "yolo.postprocess");
                return found;
            });
        }
        List<double[][]> polygons;
        List<Detection> knobs;
        long stage = System.nanoTime();
        try {
            progress.update("正在检测整图文字区域…");
            stage = System.nanoTime();
            polygons = DirectionalTextDetector.detect(bgr, settings.textDetectionMode,
                    (input, passTiming) -> detectText(input, settings.textDetectionSizeMode.longSideLimit(
                            input.cols(), input.rows(), settings.detSide), settings.boxConfidence, passTiming),
                    timing, progress);
            timing.record("ocr_detection.total", stage);
            timing.other("ocr_detection.other", "ocr_detection.total", "ocr_detection.preprocess", "ocr_detection.inference",
                    "ocr_detection.postprocess", "ocr_detection.rotation", "ocr_detection.mapping", "ocr_detection.merge");
            timing.detectedTextRegionCount = polygons.size();
            progress.update("正在汇合旋钮与文字检测结果…");
            if (yoloTask != null) {
                knobs = InferenceTasks.await(yoloTask);
                timing.copyFrom(yoloTiming);
            } else {
                stage = System.nanoTime();
                knobs = detectKnobs(bgr, settings.yoloConfidence, timing);
                timing.record("yolo.total", stage);
                timing.other("yolo.other", "yolo.total", "yolo.preprocess", "yolo.inference", "yolo.postprocess");
            }
        } finally {
            // Even if OCR or a lifecycle check fails, YOLO must finish reading bgr.
            InferenceTasks.drain(yoloTask);
        }
        timing.record("detection.wall", detectionStarted);
        progress.update("旋钮与文字检测已完成…");

        stage = System.nanoTime();
        List<Detection> regions = new ArrayList<>();
        for (int i = 0; i < polygons.size(); i++) {
            double[][] polygon = polygons.get(i);
            // Zero is an internal placeholder, never exported as an OCR confidence.
            regions.add(new Detection(i, 0, bounds(polygon), polygon, null));
        }
        timing.record("region_prepare", stage);
        progress.update("正在按坐标匹配旋钮与文字区域…");
        stage = System.nanoTime();
        MatchReport report = GreedyMatcher.matchByKnobSize(bgr.cols(), bgr.rows(), knobs, regions,
                settings.maxAngleFromDownwardDegrees, settings.searchDirection);
        List<Detection> selected = RecognitionSelection.selected(report);
        timing.record("matching", stage);

        long loopStart = System.nanoTime();
        for (int i = 0; i < selected.size(); i++) {
            Detection region = selected.get(i);
            TimingReport.TextCost cost = new TimingReport.TextCost(region.id);
            cost.textId = region.id;
            for (String key : new String[]{"crop", "angle.model_load", "angle.preprocess", "angle.inference", "angle.postprocess",
                    "recognition.preprocess", "recognition.inference", "recognition.postprocess", "filter"}) cost.timing.add(key, 0);
            long regionStart = System.nanoTime();
            progress.update("只识别已匹配区域 " + (i + 1) + "/" + selected.size() + "（整图 " + polygons.size() + " 个）…");
            Mat cropped = null;
            try {
                stage = System.nanoTime();
                try { cropped = crop(bgr, region.polygon); }
                finally { cost.timing.record("crop", stage); }
                cost.cropWidth = cropped.cols(); cost.cropHeight = cropped.rows();
                if (settings.angleClassifier) orient(cropped, cost.timing);
                TensorDecoders.Recognition recognition = recognize(cropped, cost.timing);
                stage = System.nanoTime();
                RecognitionSelection.record(region, recognition.text, recognition.confidence, settings.ocrConfidence);
                cost.timing.record("filter", stage);
            } catch (java.util.concurrent.CancellationException cancelled) {
                throw cancelled;
            } catch (Exception error) {
                RecognitionSelection.failed(region, error);
            } finally { if (cropped != null) cropped.release(); }
            cost.text = region.text == null ? "" : region.text; cost.confidence = region.confidence;
            cost.accepted = "recognized".equals(region.recognitionStatus);
            cost.recognitionStatus = region.recognitionStatus; cost.error = region.recognitionError;
            cost.timing.record("total", regionStart);
            cost.timing.other("other", "total", "crop", "angle.model_load", "angle.preprocess", "angle.inference", "angle.postprocess",
                    "recognition.preprocess", "recognition.inference", "recognition.postprocess", "filter");
            for (java.util.Map.Entry<String, Long> entry : cost.timing.stages.entrySet()) {
                if (!entry.getKey().equals("total") && !entry.getKey().equals("other")) timing.add("text_loop." + entry.getKey(), entry.getValue());
            }
            timing.textRegions.add(cost);
            timing.recognitionModelCalls += cost.timing.recognitionModelCalls;
        }
        timing.record("text_loop.total", loopStart);
        timing.other("text_loop.other", "text_loop.total", "text_loop.crop", "text_loop.angle.model_load", "text_loop.angle.preprocess",
                "text_loop.angle.inference", "text_loop.angle.postprocess", "text_loop.recognition.preprocess", "text_loop.recognition.inference",
                "text_loop.recognition.postprocess", "text_loop.filter");
        timing.record("pipeline.total", start);
        // YOLO and detection may overlap; subtract their wall span, never their sum.
        timing.other("pipeline.other", "pipeline.total", "detection.wall", "region_prepare", "matching", "text_loop.total");
        report.timing = timing;
        report.elapsedMs = timing.nanos("pipeline.total") / 1_000_000;
        return report;
    }
    private List<Detection> detectKnobs(Mat image, double confidence, TimingReport timing) throws Exception {
        long started = System.nanoTime();
        int side = models.yoloSide;
        double scale = Math.min((double) side / image.cols(), (double) side / image.rows());
        int w = (int) Math.rint(image.cols() * scale), h = (int) Math.rint(image.rows() * scale);
        int left = (int) Math.rint((side - w) / 2.0 - 0.1), top = (int) Math.rint((side - h) / 2.0 - 0.1);
        Mat resized = new Mat(), padded = new Mat();
        try {
            Imgproc.resize(image, resized, new Size(w, h), 0, 0, Imgproc.INTER_LINEAR);
            Core.copyMakeBorder(resized, padded, top, side - h - top, left, side - w - left,
                    Core.BORDER_CONSTANT, new Scalar(114, 114, 114));
            try (OnnxTensor tensor = input(padded, side, side, true, false)) {
                timing.record("yolo.preprocess", started);
                started = System.nanoTime();
                try (OrtSession.Result result = runMeasured(yolo, tensor, timing, "yolo")) {
                started = System.nanoTime();
                OnnxTensor output = (OnnxTensor) result.get(0);
                long[] shape = ((TensorInfo) output.getInfo()).getShape();
                List<Detection> found = TensorDecoders.yolo(output.getFloatBuffer(), (int) shape[1], (int) shape[2],
                        image.cols(), image.rows(), scale, left, top, confidence, 0.7);
                timing.record("yolo.postprocess", started);
                return found;
                }
            }
        } finally { resized.release(); padded.release(); }
    }
    private List<double[][]> detectText(Mat image, int limitSide, double boxThreshold, TimingReport timing) throws Exception {
        if (textEngine != null) return textEngine.detect(image, limitSide, timing);
        long started = System.nanoTime();
        int[] size = TextDetectionSizeMode.inputSize(image.cols(), image.rows(), limitSide, true);
        int w = size[0], h = size[1];
        timing.detectionSourceWidth = image.cols(); timing.detectionSourceHeight = image.rows();
        timing.detectionInputWidth = w; timing.detectionInputHeight = h;
        Mat resized = new Mat();
        try {
            Imgproc.resize(image, resized, new Size(w, h));
            try (OnnxTensor tensor = input(resized, w, h, false, true)) {
                timing.record("ocr_detection.preprocess", started);
                started = System.nanoTime();
                try (OrtSession.Result result = runMeasured(detector, tensor, timing, "ocr_detection")) {
                started = System.nanoTime();
                OnnxTensor output = (OnnxTensor) result.get(0);
                long[] shape = ((TensorInfo) output.getInfo()).getShape();
                int mapH = (int) shape[2], mapW = (int) shape[3];
                FloatBuffer buffer = output.getFloatBuffer();
                float[] probabilities = new float[mapH * mapW]; buffer.get(probabilities);
                List<double[][]> found = textBoxes(probabilities, mapW, mapH, image.cols(), image.rows(), boxThreshold);
                timing.record("ocr_detection.postprocess", started);
                return found;
                }
            }
        } finally { resized.release(); }
    }
    private List<double[][]> textBoxes(float[] probabilities, int w, int h, int destW, int destH, double threshold) {
        Mat map = new Mat(h, w, CvType.CV_32F), mask = new Mat(h, w, CvType.CV_8U), hierarchy = new Mat();
        Mat kernel = Mat.ones(2, 2, CvType.CV_8U);
        List<MatOfPoint> contours = new ArrayList<>();
        List<double[][]> boxes = new ArrayList<>();
        try {
            map.put(0, 0, probabilities);
            byte[] bitmap = new byte[w * h];
            for (int i = 0; i < bitmap.length; i++) if (probabilities[i] > 0.3f) bitmap[i] = (byte) 255;
            mask.put(0, 0, bitmap);
            Imgproc.dilate(mask, mask, kernel);
            Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            for (int i = 0; i < Math.min(1000, contours.size()); i++) {
                MatOfPoint2f contour = new MatOfPoint2f(contours.get(i).toArray());
                try {
                    RotatedRect rect = Imgproc.minAreaRect(contour);
                    if (Math.min(rect.size.width, rect.size.height) < 3) continue;
                    Point[] points = ordered(rect);
                    if (boxScore(map, points) < threshold) continue;
                    // The minimum box of a rounded offset rectangle has both dimensions enlarged by 2*d.
                    // Clipper integer quantization is not reproduced; document the resulting small differences.
                    double area = rect.size.width * rect.size.height;
                    double perimeter = 2 * (rect.size.width + rect.size.height);
                    double distance = area * 1.6 / perimeter;
                    rect.size = new Size(rect.size.width + 2 * distance, rect.size.height + 2 * distance);
                    if (Math.min(rect.size.width, rect.size.height) < 5) continue;
                    points = ordered(rect);
                    double[][] polygon = new double[4][2];
                    for (int p = 0; p < 4; p++) {
                        polygon[p][0] = TensorDecoders.clip(Math.rint(points[p].x / w * destW), destW);
                        polygon[p][1] = TensorDecoders.clip(Math.rint(points[p].y / h * destH), destH);
                    }
                    if (length(polygon[0], polygon[1]) > 3 && length(polygon[0], polygon[3]) > 3) boxes.add(polygon);
                } finally { contour.release(); }
            }
        } finally {
            map.release(); mask.release(); hierarchy.release(); kernel.release();
            for (MatOfPoint contour : contours) contour.release();
        }
        boxes.sort(Comparator.comparingDouble((double[][] box) -> box[0][1]).thenComparingDouble(box -> box[0][0]));
        for (int i = 0; i < boxes.size() - 1; i++) {
            for (int j = i; j >= 0; j--) {
                if (Math.abs(boxes.get(j + 1)[0][1] - boxes.get(j)[0][1]) < 10 && boxes.get(j + 1)[0][0] < boxes.get(j)[0][0])
                    Collections.swap(boxes, j, j + 1);
                else break;
            }
        }
        return boxes;
    }
    private double boxScore(Mat probabilities, Point[] box) {
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE, maxX = 0, maxY = 0;
        for (Point p : box) { minX = Math.min(minX, p.x); minY = Math.min(minY, p.y); maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y); }
        int x1 = Math.max(0, (int) Math.floor(minX)), y1 = Math.max(0, (int) Math.floor(minY));
        int x2 = Math.min(probabilities.cols() - 1, (int) Math.ceil(maxX)), y2 = Math.min(probabilities.rows() - 1, (int) Math.ceil(maxY));
        if (x2 < x1 || y2 < y1) return 0;
        Mat roi = probabilities.submat(new Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1));
        Mat mask = Mat.zeros(roi.rows(), roi.cols(), CvType.CV_8U);
        Point[] local = new Point[4];
        for (int i = 0; i < 4; i++) local[i] = new Point((int) (box[i].x - x1), (int) (box[i].y - y1));
        MatOfPoint polygon = new MatOfPoint(local);
        try { Imgproc.fillPoly(mask, Collections.singletonList(polygon), new Scalar(1)); return Core.mean(roi, mask).val[0]; }
        finally { roi.release(); mask.release(); polygon.release(); }
    }
    private static Point[] ordered(RotatedRect rect) {
        Point[] points = new Point[4]; rect.points(points);
        java.util.Arrays.sort(points, Comparator.comparingDouble(p -> p.x));
        Point leftTop = points[0].y < points[1].y ? points[0] : points[1];
        Point leftBottom = points[0].y < points[1].y ? points[1] : points[0];
        Point rightTop = points[2].y < points[3].y ? points[2] : points[3];
        Point rightBottom = points[2].y < points[3].y ? points[3] : points[2];
        return new Point[]{leftTop, rightTop, rightBottom, leftBottom};
    }
    private static double length(double[] a, double[] b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
    private static double[] bounds(double[][] polygon) {
        double[] box = {Double.MAX_VALUE, Double.MAX_VALUE, 0, 0};
        for (double[] p : polygon) { box[0] = Math.min(box[0], p[0]); box[1] = Math.min(box[1], p[1]); box[2] = Math.max(box[2], p[0]); box[3] = Math.max(box[3], p[1]); }
        return box;
    }
    private Mat crop(Mat image, double[][] polygon) {
        int w = Math.max(1, (int) Math.max(length(polygon[0], polygon[1]), length(polygon[2], polygon[3])));
        int h = Math.max(1, (int) Math.max(length(polygon[0], polygon[3]), length(polygon[1], polygon[2])));
        Point[] points = new Point[4]; for (int i = 0; i < 4; i++) points[i] = new Point(polygon[i][0], polygon[i][1]);
        MatOfPoint2f from = new MatOfPoint2f(points), to = new MatOfPoint2f(new Point(0, 0), new Point(w, 0), new Point(w, h), new Point(0, h));
        Mat transform = Imgproc.getPerspectiveTransform(from, to), crop = new Mat();
        try {
            Imgproc.warpPerspective(image, crop, transform, new Size(w, h), Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE, Scalar.all(0));
            if ((double) h / w >= 1.5) Core.rotate(crop, crop, Core.ROTATE_90_COUNTERCLOCKWISE);
            return crop;
        } catch (RuntimeException error) { crop.release(); throw error; }
        finally { from.release(); to.release(); transform.release(); }
    }
    private TensorDecoders.Recognition recognize(Mat crop, TimingReport timing) throws Exception {
        if (textEngine != null) return textEngine.recognize(crop, timing);
        long started = System.nanoTime();
        int height = models.ocr.recognitionHeight(), resizedWidth = Math.max(1, (int) Math.ceil(height * (double) crop.cols() / crop.rows()));
        int width = Math.max(320, resizedWidth);
        if (width > 4096) throw new IllegalArgumentException("文字区域过长，请选择范围更集中的图片");
        Mat resized = new Mat();
        try {
            Imgproc.resize(crop, resized, new Size(resizedWidth, height));
            try (OnnxTensor tensor = input(resized, width, height, false, true)) {
                timing.record("recognition.preprocess", started);
                started = System.nanoTime();
                try (OrtSession.Result result = runMeasured(recognizer, tensor, timing, "recognition")) {
                started = System.nanoTime();
                OnnxTensor output = (OnnxTensor) result.get(0);
                long[] shape = ((TensorInfo) output.getInfo()).getShape();
                TensorDecoders.Recognition found = TensorDecoders.ctc(output.getFloatBuffer(), (int) shape[1], (int) shape[2], dictionary);
                timing.record("recognition.postprocess", started);
                return found;
                }
            }
        } finally { resized.release(); }
    }
    private void orient(Mat crop, TimingReport timing) throws Exception {
        long started = System.nanoTime();
        if (classifier == null) {
            classifier = session("ch_ppocr_mobile_v2.0_cls_infer.onnx");
            timing.record("angle.model_load", started);
        }
        started = System.nanoTime();
        int width = Math.min(192, Math.max(1, (int) Math.ceil(48.0 * crop.cols() / crop.rows())));
        Mat resized = new Mat();
        try {
            Imgproc.resize(crop, resized, new Size(width, 48));
            try (OnnxTensor tensor = input(resized, 192, 48, false, true)) {
                timing.record("angle.preprocess", started);
                started = System.nanoTime();
                try (OrtSession.Result result = runMeasured(classifier, tensor, timing, "angle")) {
                started = System.nanoTime();
                FloatBuffer values = ((OnnxTensor) result.get(0)).getFloatBuffer();
                if (values.get(1) > values.get(0) && values.get(1) > 0.9f) Core.rotate(crop, crop, Core.ROTATE_180);
                timing.record("angle.postprocess", started);
                }
            }
        } finally { resized.release(); }
    }
    private OnnxTensor input(Mat image, int width, int height, boolean rgb, boolean normalize) throws OrtException {
        int imageW = image.cols(), imageH = image.rows(), plane = width * height;
        byte[] pixels = new byte[imageW * imageH * 3]; image.get(0, 0, pixels);
        float[] tensor = new float[plane * 3]; // normalized zero padding, matching RapidOCR.
        for (int y = 0; y < imageH; y++) for (int x = 0; x < imageW; x++) {
            int source = (y * imageW + x) * 3, destination = y * width + x;
            for (int channel = 0; channel < 3; channel++) {
                float value = (pixels[source + (rgb ? 2 - channel : channel)] & 255) / 255f;
                tensor[channel * plane + destination] = normalize ? (value - 0.5f) / 0.5f : value;
            }
        }
        return OnnxTensor.createTensor(environment, FloatBuffer.wrap(tensor), new long[]{1, 3, height, width});
    }
    @Override public void close() throws OrtException {
        if (textEngine != null) textEngine.close();
        OrtException first = null;
        for (OrtSession session : new OrtSession[]{classifier, recognizer, detector, yolo}) if (session != null) {
            try { session.close(); } catch (OrtException error) { if (first == null) first = error; }
        }
        classifier = recognizer = detector = yolo = null;
        if (first != null) throw first;
    }
}
