package com.example.cabinetmatcher.camera;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.util.Size;
import android.view.View;
import android.view.WindowInsets;
import android.widget.*;
import androidx.activity.ComponentActivity;
import androidx.camera.core.*;
import androidx.camera.core.resolutionselector.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.annotation.OptIn;
import androidx.camera.view.TransformExperimental;
import androidx.camera.view.transform.ImageProxyTransformFactory;
import androidx.camera.view.transform.OutputTransform;
import androidx.core.content.ContextCompat;
import com.example.cabinetmatcher.camera.core.*;
import com.google.common.util.concurrent.ListenableFuture;
import java.io.File;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.json.JSONObject;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

@OptIn(markerClass = TransformExperimental.class)
public class MainActivity extends ComponentActivity {
    private static final int CAMERA_PERMISSION = 10, EXPORT_JSON = 11, EXPORT_PNG = 12;
    // The analyzer is the OCR/coordinator worker. There is still only ONE frame in flight.
    private final ExecutorService analyzer = Executors.newSingleThreadExecutor(r -> new Thread(r, "Cabinet-OCR"));
    private final ExecutorService yoloWorker = Executors.newSingleThreadExecutor(r -> new Thread(r, "Cabinet-YOLO"));
    private final ExecutorService handWorker = Executors.newSingleThreadExecutor(r -> new Thread(r, "Cabinet-Hands"));
    private final FrameGate gate = new FrameGate();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private volatile boolean destroyed, modelsReady;
    private volatile long testDelayMs;
    private volatile CabinetPipeline.Settings activeSettings;
    private CabinetPipeline pipeline;
    private HandEngine handEngine;
    private final FrameClock handClock = new FrameClock();
    private long handSession = -1;
    private boolean handUsed;
    private double handLoadMs;
    private ProcessCameraProvider cameraProvider;
    private ImageAnalysis analysis;
    private Camera camera;
    private boolean cameraRequested, cameraBound, resumed;
    private long cameraGeneration, completed, sessionStartNs;
    private volatile FrameResult lastResult;
    private PreviewView live;
    private LiveOverlayView liveOverlay;
    private ZoomImageView resultImage;
    private MatchedRegionsView matchedRegions;
    private TextView status, modelStatus, ocrDetails, resultAge, metrics, matches, timings;
    private Button cameraButton, analysisButton, jsonButton, pngButton;
    private EditText downwardAngle, knobScore, textScore, detSide;
    private CheckBox angleClassifier;
    private Spinner resolution, yoloSize, ocrModel, searchDirection, inferenceMode, textDetectionMode, textDetectionSizeMode;
    private boolean modelSelectionEnabled, modelHasRun;
    private volatile long modelGeneration;
    private ModelChoice requestedModels;
    private File modelsDirectory;
    private double loadedModelMs, warmProcessingMs, warmYoloMs, warmOcrMs, warmHandMs;
    private long warmFrames;
    private LinearLayout settingsPanel;
    private final Runnable ageTicker = new Runnable() {
        @Override public void run() {
            FrameResult result = lastResult;
            if (!destroyed && result != null) resultAge.setText(String.format(Locale.CHINA,
                    "最近完成帧 #%d · 距收到此帧 %.1f 秒\n下方是对应历史帧，上方将旧结果叠加在实时预览。",
                    result.ticket.sequence, (System.nanoTime() - result.ticket.receivedNs) / 1e9));
            if (!destroyed && resumed) ui.postDelayed(this, 500);
        }
    };
    static final class FrameResult {
        final FrameGate.Ticket ticket;
        final MatchReport report;
        final Bitmap annotated;
        Bitmap matchingSource;
        final CabinetPipeline.Settings settings;
        final long cameraTimestampNs, finishedNs;
        final int sensorWidth, sensorHeight, rotation;
        double displayLatencyMs, modelLoadMs;
        boolean firstInferenceForModels;
        OutputTransform analysisTransform;
        float[][] hands = new float[0][];
        boolean handDetectionCompleted;
        long handTimestampMs;
        double handLoadMs;
        FrameResult(FrameGate.Ticket ticket, MatchReport report, Bitmap annotated,
                    CabinetPipeline.Settings settings, long timestamp, long finishedNs,
                    int sensorWidth, int sensorHeight, int rotation) {
            this.ticket = ticket; this.report = report; this.annotated = annotated; this.settings = settings;
            this.cameraTimestampNs = timestamp; this.finishedNs = finishedNs;
            this.sensorWidth = sensorWidth; this.sensorHeight = sensorHeight; this.rotation = rotation;
        }
        String json() throws Exception {
            JSONObject output = new JSONObject(ReportWriter.json(report));
            output.put("matching_debug",new JSONObject().put("selected_regions_kept_on_ocr_failure",true)
                    .put("all_detected_regions_drawn",true)
                    .put("raw_region_preview_available",matchingSource!=null&&!matchingSource.isRecycled())
                    .put("raw_region_preview","same-frame unannotated source; region bounding rectangle with detected polygon outline")
                    .put("recognition_threshold",settings.ocrConfidence));
            output.getJSONObject("parameters").put("inference_mode", settings.inferenceMode.name()).put("yolo_confidence", settings.yoloConfidence)
                    .put("ocr_confidence", settings.ocrConfidence)
                    .put("ocr_det_side", settings.textDetectionSizeMode == TextDetectionSizeMode.MANUAL_LIMIT ? settings.detSide : JSONObject.NULL)
                    .put("text_detection_size_mode", settings.textDetectionSizeMode.name())
                    .put("angle_classifier", settings.angleClassifier)
                    .put("text_detection_mode", settings.textDetectionMode.name())
                    .put("yolo_input_size", settings.models.yoloSide).put("ocr_model", settings.models.ocr.name());
            output.put("models", new JSONObject().put("label", settings.models.label())
                    .put("execution_provider", settings.models.ocr.usesMlKit() ? "YOLO ONNX CPU + ML Kit bundled" : "CPUExecutionProvider")
                    .put("yolo_execution_provider", "CPUExecutionProvider").put("ocr_backend", settings.models.ocr.backend())
                    .put("checkpoint", "train9/weights/best.pt")
                    .put("yolo_file", settings.models.yoloFile()).put("yolo_input_size", settings.models.yoloSide)
                    .put("ocr_model", settings.models.ocr.name()).put("ocr_detection_file", settings.models.ocr.usesMlKit() ? JSONObject.NULL : settings.models.ocr.detectionFile)
                    .put("ocr_recognition_file", settings.models.ocr.usesMlKit() ? JSONObject.NULL : settings.models.ocr.recognitionFile)
                    .put("dictionary_file", settings.models.ocr.usesMlKit() ? JSONObject.NULL : settings.models.ocr.dictionaryFile)
                    .put("dictionary_classes", settings.models.ocr.usesMlKit() ? JSONObject.NULL : settings.models.ocr.classes)
                    .put("recognition_input_height", settings.models.ocr.usesMlKit() ? JSONObject.NULL : settings.models.ocr.recognitionHeight())
                    .put("load_ms_outside_frame", modelLoadMs).put("first_inference_for_loaded_models", firstInferenceForModels));
            output.put("ocr_engine", new JSONObject().put("backend", settings.models.ocr.backend())
                    .put("whole_image_sdk_reads_text", settings.models.ocr.usesMlKit())
                    .put("whole_image_sdk_calls", settings.models.ocr.usesMlKit() ? report.timing.detectionPasses.size() : 0)
                    .put("whole_image_strings_used_for_matching", false)
                    .put("selected_crops_recognized_after_matching", true)
                    .put("selected_crop_calls", report.timing.recognitionModelCalls)
                    .put("detection_box_threshold_applied", !settings.models.ocr.usesMlKit())
                    .put("confidence_source", settings.models.ocr.usesMlKit() ? "SDK line confidence weighted by character count" : "CTC mean retained-character probability"));
            if (settings.models.ocr.usesMlKit()) {
                output.put("algorithm", "ML Kit whole-image text-line locations; knob-confidence-first geometry matching; re-recognize assigned crops");
                output.getJSONObject("models").put("ocr_sdk_artifact", "com.google.mlkit:text-recognition-chinese:16.0.1");
            }
            output.put("hand_landmarks", HandJson.write(hands, report.width, report.height, handDetectionCompleted,
                    ticket.sequence, cameraTimestampNs, handTimestampMs));
            if (settings.inferenceMode == InferenceMode.PARALLEL) {
                output.put("execution_order", new org.json.JSONArray(new String[]{"parallel_yolo_ocr_detection_and_hands",
                        "join_yolo_and_ocr_detection", "geometry_matching", "selected_region_recognition",
                        "join_cabinet_and_hands", "combined_annotation"}));
            } else {
                if (handDetectionCompleted) output.getJSONArray("execution_order").put("hand_landmarks_same_frame");
                output.getJSONArray("execution_order").put("combined_annotation");
            }
            output.put("execution", new JSONObject().put("mode", settings.inferenceMode.name())
                    .put("same_frame_only", true).put("all_branches_joined_before_publish", handDetectionCompleted)
                    .put("max_concurrent_model_tasks", settings.inferenceMode == InferenceMode.PARALLEL ? 3 : 1)
                    .put("ort_intra_op_threads_per_session", 2).put("ort_inter_op_threads_per_session", 1)
                    .put("ocr_recognition_requires", "YOLO and whole-image text detection, then geometry matching")
                    .put("hand_inference_independent_of_cabinet", true));
            output.getJSONObject("models").put("hand_model", "hand_landmarker.task").put("hand_backend", "MediaPipe CPU")
                    .put("hand_model_load_ms_outside_frame", handLoadMs);
            output.getJSONObject("timings").put("scope", "Durations are wall-clock spans, not CPU time. In PARALLEL, model durations overlap and MUST NOT be added. detection.wall spans YOLO + text detection including join; pipeline.total is cabinet path including joins/matching/selected OCR; hands.total includes its own bitmap conversion and optional tracker reset; combined.total joins cabinet + hands and includes snapshot/annotations; camera.processing also includes frame conversion; excludes export and screen presentation");
            output.put("coordinate_system", "upright cropped analysis frame pixels; top-left origin; x right; y down");
            output.put("video", new JSONObject().put("frame_id", ticket.sequence).put("session_id", ticket.generation)
                    .put("camera_timestamp_ns", cameraTimestampNs).put("received_monotonic_ns", ticket.receivedNs)
                    .put("processing_ms", (finishedNs - ticket.receivedNs) / 1e6)
                    .put("receive_to_ui_submission_ms", displayLatencyMs)
                    .put("sensor_frame_width", sensorWidth).put("sensor_frame_height", sensorHeight)
                    .put("rotation_degrees", rotation).put("backpressure_strategy", "KEEP_ONLY_LATEST")
                    .put("max_concurrent_frames", 1)
                    .put("max_concurrent_inferences", settings.inferenceMode == InferenceMode.PARALLEL ? 3 : 1).put("result_overlay_target", "live_preview_and_processed_frame")
                    .put("live_overlay", new JSONObject().put("mode", "latest_completed_result_over_live_preview")
                            .put("motion_compensation", false).put("tracking", false)
                            .put("age_reference", "analysis_frame_received_monotonic_ns")
                            .put("coordinate_mapping", "CameraX shared ViewPort; cropped rotated ImageProxy to PreviewView OutputTransform")));
            return output.toString(2);
        }
        void recycleImages() {
            if(annotated!=null&&!annotated.isRecycled())annotated.recycle();
            if(matchingSource!=null&&matchingSource!=annotated&&!matchingSource.isRecycled())matchingSource.recycle();
        }
    }
    @Override public void onCreate(Bundle state) {
        super.onCreate(state); buildUi();
        android.content.SharedPreferences preferences = getPreferences(MODE_PRIVATE);
        yoloSize.setSelection(preferences.getInt("yoloSide", 1280) == 640 ? 0 : 1);
        ocrModel.setSelection(ModelChoice.Ocr.fromStoredId(preferences.getString("ocrModel", null),
                preferences.getBoolean("ocrV6", false)).ordinal());
        inferenceMode.setSelection(preferences.getBoolean("parallel", false) ? 1 : 0);
        textDetectionMode.setSelection(TextDetectionMode.fromStoredId(preferences.getString("textDetectionMode", null)).ordinal());
        textDetectionSizeMode.setSelection(TextDetectionSizeMode.fromStoredId(preferences.getString("textDetectionSizeMode", null)).ordinal());
        if (state != null) {
            yoloSize.setSelection(state.getInt("yoloChoice", yoloSize.getSelectedItemPosition()));
            int oldChoice = state.getInt("ocrChoice", ocrModel.getSelectedItemPosition());
            ModelChoice.Ocr fallback = oldChoice >= 0 && oldChoice < ModelChoice.Ocr.values().length
                    ? ModelChoice.Ocr.values()[oldChoice] : ModelChoice.Ocr.V4;
            ocrModel.setSelection(ModelChoice.Ocr.fromStoredId(state.getString("ocrModelId", fallback.name()), false).ordinal());
            downwardAngle.setText(state.getString("angle", "45")); knobScore.setText(state.getString("knob", "0.25"));
            textScore.setText(state.getString("text", "0.5")); detSide.setText(state.getString("detSide", "1600"));
            angleClassifier.setChecked(state.getInt("settingsVersion", 0) >= 4 ? state.getBoolean("classifier", true) : true); resolution.setSelection(state.getInt("resolution", 0));
            searchDirection.setSelection(state.getInt("searchDirection", 0));
            inferenceMode.setSelection(state.getInt("inferenceMode", inferenceMode.getSelectedItemPosition()));
            textDetectionMode.setSelection(TextDetectionMode.fromStoredId(state.getString("textDetectionMode",
                    TextDetectionMode.values()[textDetectionMode.getSelectedItemPosition()].name())).ordinal());
            textDetectionSizeMode.setSelection(TextDetectionSizeMode.fromStoredId(state.getString("textDetectionSizeMode",
                    TextDetectionSizeMode.values()[textDetectionSizeMode.getSelectedItemPosition()].name())).ordinal());
        }
        textDetectionMode.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                getPreferences(MODE_PRIVATE).edit().putString("textDetectionMode", TextDetectionMode.values()[position].name()).apply();
            }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });
        textDetectionSizeMode.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                getPreferences(MODE_PRIVATE).edit().putString("textDetectionSizeMode", TextDetectionSizeMode.values()[position].name()).apply();
                updateButtons();
            }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });
        updateButtons();
        if (!OpenCVLoader.initLocal()) { status.setText("OpenCV 初始化失败，请安装与设备架构一致的 APK。"); cameraButton.setEnabled(false); return; }
        modelSelectionEnabled = true;
        AdapterView.OnItemSelectedListener modelListener = new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (modelSelectionEnabled) requestModels(selectedModels());
            }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        };
        yoloSize.setOnItemSelectedListener(modelListener); ocrModel.setOnItemSelectedListener(modelListener);
        requestModels(selectedModels());
    }
    private ModelChoice selectedModels() {
        return new ModelChoice(yoloSize.getSelectedItemPosition() == 0 ? 640 : 1280,
                ModelChoice.Ocr.values()[ocrModel.getSelectedItemPosition()]);
    }
    private void requestModels(ModelChoice choice) {
        if (destroyed || choice.equals(requestedModels)) return;
        stopAnalysis(); modelsReady = false; requestedModels = choice; long generation = ++modelGeneration;
        liveOverlay.clear(); liveOverlay.state("模型切换中，等待新结果");
        getPreferences(MODE_PRIVATE).edit().putInt("yoloSide", choice.yoloSide)
                .putString("ocrModel", choice.ocr.name()).apply();
        ocrDetails.setText(choice.ocr.description);
        modelStatus.setText("正在加载：" + choice.label() + "\n加载期间可继续预览，完成后点击开始识别。"); updateButtons();
        // Inference, session replacement and close share one executor: no session is closed in use.
        analyzer.execute(() -> {
            if (destroyed || generation != modelGeneration) return;
            CabinetPipeline replacement = null;
            try {
                if (modelsDirectory == null) modelsDirectory = ModelAssets.prepare(getApplicationContext());
                if (destroyed || generation != modelGeneration) return;
                if (pipeline != null) { pipeline.close(); pipeline = null; }
                replacement = CabinetPipelines.create(modelsDirectory, choice);
                if (destroyed || generation != modelGeneration) { replacement.close(); return; }
                if (handEngine == null) {
                    InferenceTasks.await(handWorker.submit(() -> {
                        long handStarted = System.nanoTime(); handEngine = new HandEngine(getApplicationContext());
                        handLoadMs = (System.nanoTime() - handStarted) / 1e6; handUsed = false;
                        return null;
                    }));
                }
                if (destroyed || generation != modelGeneration) { replacement.close(); return; }
                pipeline = replacement; modelHasRun = false;
                loadedModelMs = pipeline.initializationTiming.millis("model_load.total");
                final double loadMs = loadedModelMs;
                runOnUiThread(() -> {
                    if (!destroyed && generation == modelGeneration) {
                        modelsReady = true;
                        liveOverlay.state(cameraBound ? "实时预览 · 等待开始识别" : "模型已就绪 · 等待开启摄像头");
                        modelStatus.setText(String.format(Locale.CHINA, "已就绪：%s + 手部关键点\n柜体模型加载 %.0f ms · 手部加载 %.0f ms（均在帧外）", choice.label(), loadMs, handLoadMs));
                        updateButtons();
                    }
                });
            } catch (Exception | OutOfMemoryError error) {
                if (replacement != null) try { replacement.close(); } catch (Exception ignored) {}
                runOnUiThread(() -> { if (!destroyed && generation == modelGeneration) {
                    modelStatus.setText("模型加载失败：" + error.getMessage() + "\n可选择其他组合后重试。"); updateButtons();
                    liveOverlay.state("模型加载失败");
                } });
            }
        });
    }
    private void buildUi() {
        ScrollView scroll = new ScrollView(this); scroll.setFillViewport(true); scroll.setBackgroundColor(Color.rgb(244, 247, 250));
        LinearLayout root = new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setPadding(dp(14), dp(12), dp(14), dp(24));
        scroll.addView(root); setContentView(scroll);
        if (Build.VERSION.SDK_INT >= 30) scroll.setOnApplyWindowInsetsListener((view, insets) -> {
            android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars());
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom); return insets;
        });
        root.addView(label("开关柜 · 摄像头实验版", 23)); root.addView(label("实时预览持续播放；识别完成后再处理最新帧。", 14));
        modelStatus = label("正在准备本机模型…", 13); root.addView(modelStatus);
        root.addView(label("暂停识别后选择模型（" + ModelChoice.Ocr.values().length + " 种 OCR × 2 种 YOLO 尺寸）", 16));
        yoloSize = new Spinner(this); yoloSize.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, new String[]{"YOLO 输入：640 × 640", "YOLO 输入：1280 × 1280"}));
        yoloSize.setSelection(1); root.addView(yoloSize);
        String[] ocrLabels = new String[ModelChoice.Ocr.values().length];
        for (ModelChoice.Ocr ocr : ModelChoice.Ocr.values()) ocrLabels[ocr.ordinal()] = "OCR：" + ocr.label;
        ocrModel = new Spinner(this); ocrModel.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, ocrLabels)); root.addView(ocrModel);
        ocrDetails = label("所有 OCR 均在本机离线运行；请保持相同画面和参数比较。", 12); root.addView(ocrDetails);
        root.addView(label("文字检测方向（暂停识别后切换）", 15));
        String[] detectionLabels = new String[TextDetectionMode.values().length];
        for (TextDetectionMode mode : TextDetectionMode.values()) detectionLabels[mode.ordinal()] = mode.label;
        textDetectionMode = new Spinner(this);
        textDetectionMode.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, detectionLabels));
        root.addView(textDetectionMode);
        root.addView(label("四方向可补检侧转、倒置文字，合并重复区域；耗时会增加。此项不改变旋钮的上/下/左/右匹配方向。", 12));
        root.addView(label("视频采集尺寸与 YOLO 输入尺寸独立；下方设置可选择文字检测跟随实际分析帧尺寸。", 12));
        root.addView(label("执行模式（暂停后切换；同一帧三项完成后才更新）", 15));
        inferenceMode = new Spinner(this); inferenceMode.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, new String[]{InferenceMode.SERIAL.label, InferenceMode.PARALLEL.label}));
        root.addView(inferenceMode);
        root.addView(label("并行任务耗时有重叠；请比较整帧处理时间和后续帧均值。", 12));
        LinearLayout actions = new LinearLayout(this); root.addView(actions);
        cameraButton = button("开启摄像头"); analysisButton = button("开始识别");
        actions.addView(cameraButton, weight()); actions.addView(analysisButton, weight());
        cameraButton.setOnClickListener(v -> { if (cameraRequested) closeCamera(); else requestCamera(); });
        analysisButton.setOnClickListener(v -> { if (gate.isRunning()) stopAnalysis(); else startAnalysis(); });
        root.addView(label("实时预览＋旋钮文本匹配＋手部骨架", 16));
        live = new PreviewView(this); live.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        live.setScaleType(PreviewView.ScaleType.FIT_CENTER); live.setBackgroundColor(Color.BLACK);
        FrameLayout previewStack = new FrameLayout(this);
        previewStack.addView(live, new FrameLayout.LayoutParams(-1, -1));
        liveOverlay = new LiveOverlayView(this, live);
        previewStack.addView(liveOverlay, new FrameLayout.LayoutParams(-1, -1));
        root.addView(previewStack, new LinearLayout.LayoutParams(-1, dp(420)));
        live.setOnTouchListener((view, event) -> {
            if (event.getAction() == android.view.MotionEvent.ACTION_UP && camera != null) {
                MeteringPoint point = live.getMeteringPointFactory().createPoint(event.getX(), event.getY());
                camera.getCameraControl().startFocusAndMetering(new FocusMeteringAction.Builder(point).build()); view.performClick();
            }
            return true;
        });
        status = label("先开启摄像头，再开始识别。", 14); root.addView(status); metrics = label("尚未处理视频帧", 14); root.addView(metrics);
        root.addView(label("最近完成帧 · 同色框和箭头表示匹配", 17)); resultAge = label("等待第一帧结果", 13); root.addView(resultAge);
        resultImage = new ZoomImageView(this); root.addView(resultImage, new LinearLayout.LayoutParams(-1, dp(300)));
        root.addView(label("匹配区域原图预览 · K旋钮 → R文字框",17));
        root.addView(label("橙色也表示已经选中文本框，只是 OCR 未通过；灰框是未选中区域。点击“暂停识别”可固定当前结果，下面逐项查看原始区域、置信度和未匹配原因。",12));
        matchedRegions=new MatchedRegionsView(this);root.addView(matchedRegions,new LinearLayout.LayoutParams(-1,-2));
        matches = label("匹配结果将在这里更新", 14); root.addView(matches);
        LinearLayout export = new LinearLayout(this); root.addView(export); jsonButton = button("导出该帧 JSON"); pngButton = button("导出该帧示意图");
        export.addView(jsonButton, weight()); export.addView(pngButton, weight()); jsonButton.setOnClickListener(v -> export(EXPORT_JSON)); pngButton.setOnClickListener(v -> export(EXPORT_PNG));
        settingsPanel = new LinearLayout(this); settingsPanel.setOrientation(LinearLayout.VERTICAL); root.addView(settingsPanel);
        settingsPanel.addView(label("暂停识别后可调整设置", 17));
        searchDirection = new Spinner(this);
        searchDirection.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item,
                new String[]{"匹配方向：下方", "匹配方向：上方", "匹配方向：左侧", "匹配方向：右侧", "匹配方向：四向同时"}));
        settingsPanel.addView(searchDirection);
        settingsPanel.addView(label("四向按最近可用文本匹配；半角 45° 时覆盖周围全部方向。", 12));
        resolution = new Spinner(this);
        resolution.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item,
                new String[]{"视频分析优先 1920×1080", "视频分析优先 1280×720", "视频分析优先 640×480"}));
        settingsPanel.addView(resolution);
        resolution.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> parent, View view, int position, long id) { if (cameraRequested) { closeCamera(); requestCamera(); } }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });
        settingsPanel.addView(label("匹配范围：中心距离 ≤ 当前旋钮框对角线；文本框至少 1/3 面积在当前旋钮框外。",13));
        downwardAngle = field("搜索扇区半角（0～60°）", "45"); knobScore = field("旋钮置信度", "0.25");
        textScore = field("OCR 识别置信度", "0.5");
        settingsPanel.addView(label("文字检测尺寸（暂停识别后切换）",15));
        String[] sizeLabels = new String[TextDetectionSizeMode.values().length];
        for (TextDetectionSizeMode mode : TextDetectionSizeMode.values()) sizeLabels[mode.ordinal()] = mode.label;
        textDetectionSizeMode = new Spinner(this);
        textDetectionSizeMode.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,sizeLabels));
        settingsPanel.addView(textDetectionSizeMode);
        settingsPanel.addView(label("跟随模式保留相机实际分析帧的分辨率。高分辨率与四方向检测组合会增加耗时和内存占用。",12));
        detSide = field("OCR 手动检测长边（32～2048，仅手动模式生效）", "1600");
        angleClassifier = new CheckBox(this); angleClassifier.setText("启用 0/180° 文字方向纠正（默认开启）"); angleClassifier.setChecked(true); settingsPanel.addView(angleClassifier);
        timings = label("每步耗时将在处理完成后显示。", 13); root.addView(timings); updateButtons();
    }
    private void requestCamera() {
        if (!resumed || destroyed) return;
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) { requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION); return; }
        cameraRequested = true; long generation = ++cameraGeneration; status.setText("正在开启后置摄像头…");
        liveOverlay.clear(); liveOverlay.state("正在开启摄像头"); updateButtons();
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            if (!cameraRequested || generation != cameraGeneration || destroyed || !resumed) return;
            try {
                cameraProvider = future.get(); cameraProvider.unbindAll();
                if (!cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) throw new IllegalStateException("未找到后置摄像头");
                int selection = resolution.getSelectedItemPosition(); Size preferred = selection == 0 ? new Size(1920, 1080) : selection == 1 ? new Size(1280, 720) : new Size(640, 480);
                ResolutionSelector sizes = new ResolutionSelector.Builder()
                        .setAspectRatioStrategy(selection == 2 ? AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY : AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(new ResolutionStrategy(preferred, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build();
                Preview preview = new Preview.Builder().setResolutionSelector(sizes).build(); preview.setSurfaceProvider(live.getSurfaceProvider());
                analysis = new ImageAnalysis.Builder().setResolutionSelector(sizes).setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build(); analysis.setAnalyzer(analyzer, this::analyzeFrame);
                ViewPort viewPort = live.getViewPort();
                if (viewPort == null) throw new IllegalStateException("预览布局尚未就绪，请再次开启摄像头");
                UseCaseGroup group = new UseCaseGroup.Builder().setViewPort(viewPort).addUseCase(preview).addUseCase(analysis).build();
                camera = cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, group);
                liveOverlay.state("实时预览 · 等待开始识别");
                cameraBound = true; status.setText("预览已开启，点击“开始识别”。实际视频尺寸以结果为准。"); updateButtons();
            } catch (Exception error) { closeCamera(); status.setText("摄像头启动失败：" + error.getMessage()); }
        }, ContextCompat.getMainExecutor(this));
    }
    @Override public void onRequestPermissionsResult(int request, String[] permissions, int[] grants) {
        super.onRequestPermissionsResult(request, permissions, grants);
        if (request == CAMERA_PERMISSION) {
            if (grants.length > 0 && grants[0] == PackageManager.PERMISSION_GRANTED) requestCamera();
            else status.setText("需要相机权限才能预览；可在系统应用设置中允许相机权限。");
        }
    }
    private void closeCamera() {
        cameraRequested = false; cameraBound = false; cameraGeneration++; stopAnalysis();
        liveOverlay.clear(); liveOverlay.state("摄像头已关闭");
        if (analysis != null) { analysis.clearAnalyzer(); analysis = null; } if (cameraProvider != null) cameraProvider.unbindAll(); camera = null;
        status.setText("摄像头已关闭，最近完成帧仍可查看和导出。"); updateButtons();
    }
    private void startAnalysis() {
        if (!modelsReady || !cameraBound) return; CabinetPipeline.Settings settings = new CabinetPipeline.Settings();
        try {
            settings.models = selectedModels();
            settings.inferenceMode = InferenceMode.values()[inferenceMode.getSelectedItemPosition()];
            settings.searchDirection = SearchDirection.values()[searchDirection.getSelectedItemPosition()];
            settings.textDetectionMode = TextDetectionMode.values()[textDetectionMode.getSelectedItemPosition()];
            settings.textDetectionSizeMode = TextDetectionSizeMode.values()[textDetectionSizeMode.getSelectedItemPosition()];
            if (!settings.models.equals(requestedModels)) return;
            settings.maxAngleFromDownwardDegrees = Double.parseDouble(downwardAngle.getText().toString());
            settings.yoloConfidence = Double.parseDouble(knobScore.getText().toString()); settings.ocrConfidence = Double.parseDouble(textScore.getText().toString());
            if (settings.textDetectionSizeMode == TextDetectionSizeMode.MANUAL_LIMIT)
                settings.detSide = Integer.parseInt(detSide.getText().toString());
            settings.angleClassifier = angleClassifier.isChecked(); settings.validate();
        } catch (IllegalArgumentException error) { status.setText("请检查设置：" + error.getMessage()); return; }
        getPreferences(MODE_PRIVATE).edit().putBoolean("parallel", settings.inferenceMode == InferenceMode.PARALLEL).apply();
        activeSettings = settings; completed = 0; warmFrames = 0; warmProcessingMs = 0; warmYoloMs = 0; warmOcrMs = 0; warmHandMs = 0; sessionStartNs = System.nanoTime(); gate.start();
        liveOverlay.clear(); liveOverlay.state("识别中 · 同帧柜体＋手部结果");
        status.setText("识别中：框和文字来自最近完成帧，移动相机可观察滞后。"); updateButtons();
    }
    private void stopAnalysis() {
        boolean wasRunning = gate.isRunning(); gate.stop();
        if (liveOverlay != null) liveOverlay.state("识别已暂停 · 结果保持不更新");
        if (wasRunning) status.setText("识别已暂停，实时预览继续；最近完成帧保留。"); updateButtons();
    }
    private void analyzeFrame(ImageProxy image) {
        FrameGate.Ticket ticket = gate.tryBegin(System.nanoTime());
        if (ticket == null || destroyed) { if (ticket != null) gate.finish(ticket); image.close(); return; }
        CabinetPipeline.Settings settings = activeSettings; Mat bgr = null; Bitmap source = null, annotated = null;
        Future<HandResult> handTask = null;
        try {
            long timestamp = image.getImageInfo().getTimestamp(); int sensorWidth = image.getWidth(), sensorHeight = image.getHeight();
            ImageProxyTransformFactory transformFactory = new ImageProxyTransformFactory();
            transformFactory.setUsingCropRect(true); transformFactory.setUsingRotationDegrees(true);
            OutputTransform analysisTransform = transformFactory.getOutputTransform(image);
            int rotation = image.getImageInfo().getRotationDegrees(); long stage = System.nanoTime(); bgr = CameraFrames.bgr(image); long conversionNs = System.nanoTime() - stage;
            // Instrumentation can simulate a slow model; production leaves this at zero.
            if (testDelayMs > 0) Thread.sleep(testDelayMs);
            long combinedStarted = System.nanoTime();
            final Mat frame = bgr;
            if (settings.inferenceMode == InferenceMode.PARALLEL)
                handTask = handWorker.submit(() -> detectHands(frame, ticket));
            MatchReport found = pipeline.run(bgr, settings, message -> {
                if (destroyed || !gate.isCurrent(ticket)) throw new CancellationException("Stopped video session");
            }, yoloWorker);
            if (handTask == null) handTask = handWorker.submit(() -> detectHands(frame, ticket));
            HandResult hand = InferenceTasks.await(handTask);
            found.timing.copyFrom(hand.timing);
            found.imageName = "camera_frame_" + ticket.sequence; found.timing.modelsReused = true; found.timing.add("camera.convert", conversionNs);
            if (destroyed || !gate.isCurrent(ticket)) throw new CancellationException("Stopped before hand detection");
            stage = System.nanoTime(); source = CameraFrames.bitmap(bgr); found.timing.record("camera.bitmap", stage);
            stage = System.nanoTime(); annotated = Annotations.draw(source, found); found.timing.record("annotation.base", stage);
            float[][] hands = hand.hands;
            if (destroyed || !gate.isCurrent(ticket)) throw new CancellationException("Stopped before annotation");
            stage = System.nanoTime();
            HandDrawing.draw(new android.graphics.Canvas(annotated), hands, found.width, found.height,
                    new android.graphics.Matrix(), Math.max(1, found.width / 550f));
            found.timing.record("annotation.hands", stage);
            found.timing.add("annotation", found.timing.nanos("annotation.base") + found.timing.nanos("annotation.hands"));
            found.timing.record("combined.total", combinedStarted); found.timing.record("camera.processing", ticket.receivedNs);
            FrameResult result = new FrameResult(ticket, found, annotated, settings, timestamp, System.nanoTime(), sensorWidth, sensorHeight, rotation);
            result.modelLoadMs = loadedModelMs; result.firstInferenceForModels = !modelHasRun; modelHasRun = true;
            result.analysisTransform = analysisTransform;
            result.hands = hands; result.handDetectionCompleted = true; result.handTimestampMs = hand.timestamp; result.handLoadMs = handLoadMs;
            result.matchingSource=source;source=null; // The published frame owns the raw image used for region previews.
            annotated = null; runOnUiThread(() -> publish(result));
        } catch (CancellationException ignored) {
            // Lifecycle or pause invalidated this frame; never publish it.
        } catch (Exception | OutOfMemoryError error) {
            runOnUiThread(() -> { if (!destroyed && gate.isCurrent(ticket)) { stopAnalysis(); status.setText("识别已暂停：" + error.getMessage()); } });
        } finally {
            // A failed/cancelled cabinet branch cannot release the frame under native hand inference.
            InferenceTasks.drain(handTask);
            if (source != null && !source.isRecycled()) source.recycle(); if (annotated != null) annotated.recycle(); if (bgr != null) bgr.release(); gate.finish(ticket);
            // Hold this proxy THROUGH inference so the next callback receives the newest frame.
            // Closing early can leave an old analyzer callback waiting in the executor.
            image.close();
        }
    }
    private static final class HandResult {
        final TimingReport timing = new TimingReport();
        float[][] hands;
        long timestamp;
    }
    /** Runs exclusively on handWorker, including tracker creation/reset/close. */
    private HandResult detectHands(Mat frame, FrameGate.Ticket ticket) throws Exception {
        if (destroyed || !gate.isCurrent(ticket)) throw new CancellationException("Stopped before hand detection");
        HandResult result = new HandResult(); long started = System.nanoTime(); Bitmap input = null;
        try {
            result.timing.add("hands.model_reset", 0);
            if (handSession != ticket.generation) {
                if (handUsed) { long stage = System.nanoTime(); handEngine.reset(); result.timing.record("hands.model_reset", stage); }
                handSession = ticket.generation; handUsed = true;
            }
            long stage = System.nanoTime(); input = CameraFrames.bitmap(frame); result.timing.record("hands.bitmap", stage);
            if (destroyed || !gate.isCurrent(ticket)) throw new CancellationException("Stopped during hand setup");
            result.timestamp = handClock.next(SystemClock.uptimeMillis());
            stage = System.nanoTime(); result.hands = handEngine.detect(input, result.timestamp);
            result.timing.record("hands.inference", stage);
            return result;
        } finally {
            if (input != null && !input.isRecycled()) input.recycle();
            result.timing.record("hands.total", started);
        }
    }
    private void publish(FrameResult result) {
        if (destroyed || !resumed || !gate.isCurrent(result.ticket)) { result.recycleImages(); return; }
        result.displayLatencyMs = (System.nanoTime() - result.ticket.receivedNs) / 1e6;
        FrameResult previous = lastResult; lastResult = result; completed++; resultImage.show(result.annotated, true);
        matchedRegions.show(result.report,result.matchingSource,result.settings.ocrConfidence);
        if (previous != null) resultImage.post(previous::recycleImages);
        MatchReport found = result.report; double seconds = (System.nanoTime() - sessionStartNs) / 1e9;
        liveOverlay.show(result, completed / Math.max(seconds, 0.001));
        if (completed > 1) {
            warmFrames++; warmProcessingMs += (result.finishedNs - result.ticket.receivedNs) / 1e6;
            warmYoloMs += found.timing.millis("yolo.total");
            warmOcrMs += found.timing.millis("ocr_detection.total") + found.timing.millis("text_loop.total");
            warmHandMs += found.timing.millis("hands.total");
        }
        String warm = warmFrames == 0 ? "本次首帧；后续显示去除首帧的均值。" : String.format(Locale.CHINA,
                "后续 %d 帧均值：处理 %.0f ms · YOLO %.0f ms · OCR %.0f ms · 手部 %.0f ms", warmFrames,
                warmProcessingMs / warmFrames, warmYoloMs / warmFrames, warmOcrMs / warmFrames, warmHandMs / warmFrames);
        metrics.setText(result.settings.inferenceMode.label + "\n" + result.settings.models.label() + " · 方向 " + result.settings.searchDirection.label + " · 手 " + result.hands.length + "\n文字检测：" + result.settings.textDetectionMode.label + "\n" + (result.firstInferenceForModels ? "模型加载后的首次推理\n" : "模型已复用\n") + String.format(Locale.CHINA,
                "本次已完成 %d 帧 · 平均更新 %.2f 帧/秒\n处理帧 %d×%d · 本帧处理 %.0f ms\n收到帧→界面提交 %.0f ms（不含曝光和屏幕实际绘帧）",
                completed, completed / Math.max(seconds, 0.001), found.width, found.height,
                (result.finishedNs - result.ticket.receivedNs) / 1e6, result.displayLatencyMs) + "\n" + warm);
        StringBuilder list = new StringBuilder(String.format(Locale.CHINA,
                "旋钮 %d · 文字区域 %d · 匹配 %d · 文字通过 %d\n识别网络 %d 次 · 跳过 %d 个区域\n",
                found.knobs.size(), found.texts.size(), found.matches.size(), found.recognizedMatches(), found.timing.recognitionModelCalls, found.unmatchedTexts.size()));
        for (MatchReport.Decision decision : found.decisions) {
            if (decision.pair == null) list.append("K").append(decision.knob.id).append(" 未匹配：").append(GreedyMatcher.describe(decision.reason)).append('\n');
            else list.append("K").append(decision.knob.id).append(" → R").append(decision.pair.text.id).append("：")
                    .append(decision.pair.text.text == null ? "（未读出）" : decision.pair.text.text).append(" · ").append(RecognitionLabel.describe(decision.pair.text)).append('\n');
        }
        list.append("同帧检测到 ").append(result.hands.length).append(" 只手，黄色点为食指指尖。\n");
        matches.setText(list); timings.setText(result.settings.inferenceMode.label + "\n并行耗时重叠，请以整帧处理/后续帧均值比较。\n" + String.format(Locale.CHINA,
                "视频帧转换：%.3f ms\n同帧位图转换：%.3f ms\n帧上标注：%.3f ms\nYOLO 合计：%.3f ms\n文字检测合计：%.3f ms\n匹配：%.3f ms\n匹配区域识别合计：%.3f ms\n柜体检测＋OCR＋匹配合计：%.3f ms\n模型在开启识别前已准备，连续帧复用。\n\n",
                found.timing.millis("camera.convert"), found.timing.millis("camera.bitmap"), found.timing.millis("annotation"), found.timing.millis("yolo.total"),
                found.timing.millis("ocr_detection.total"), found.timing.millis("matching"),
                found.timing.millis("text_loop.total"), found.timing.millis("pipeline.total")) + String.format(Locale.CHINA,
                "手部跟踪器重置：%.3f ms\n手部检测＋关键点整理：%.3f ms\n手部合计：%.3f ms\n同帧柜体＋手部＋位图/标注合计：%.3f ms\n\n",
                found.timing.millis("hands.model_reset"), found.timing.millis("hands.inference"),
                found.timing.millis("hands.total"), found.timing.millis("combined.total")) + TimingDisplay.details(found.timing));
        status.setText("持续识别中 · 实时画面上的框按推理完成频率更新；移动相机可观察滞后。");
        ui.removeCallbacks(ageTicker); ageTicker.run(); updateButtons();
    }
    private void export(int type) {
        if (lastResult == null) return; stopAnalysis();
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
                .setType(type == EXPORT_JSON ? "application/json" : "image/png")
                .putExtra(Intent.EXTRA_TITLE, "camera_frame_" + lastResult.ticket.sequence + (type == EXPORT_JSON ? ".json" : ".png"));
        try { startActivityForResult(intent, type); } catch (RuntimeException error) { status.setText("导出失败：" + error.getMessage()); }
    }
    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data); FrameResult frame = lastResult;
        if (result != RESULT_OK || data == null || data.getData() == null || frame == null || (request != EXPORT_JSON && request != EXPORT_PNG)) return;
        try (OutputStream output = getContentResolver().openOutputStream(data.getData(), "wt")) {
            if (output == null) throw new IllegalStateException("无法写入所选位置");
            if (request == EXPORT_JSON) output.write(frame.json().getBytes(StandardCharsets.UTF_8));
            else if (!frame.annotated.compress(Bitmap.CompressFormat.PNG, 100, output)) throw new IllegalStateException("图片保存失败");
            status.setText("该帧结果已保存；可重新开启摄像头继续测试。");
        } catch (Exception error) { status.setText("导出失败：" + error.getMessage()); }
    }
    private void updateButtons() {
        if (cameraButton == null) return;
        cameraButton.setText(cameraRequested ? "关闭摄像头" : "开启摄像头");
        analysisButton.setText(gate.isRunning() ? "暂停识别" : "开始识别"); analysisButton.setEnabled(modelsReady && cameraBound);
        if (yoloSize != null) { yoloSize.setEnabled(!gate.isRunning()); ocrModel.setEnabled(!gate.isRunning()); inferenceMode.setEnabled(!gate.isRunning()); }
        if (textDetectionMode != null) textDetectionMode.setEnabled(!gate.isRunning());
        if (settingsPanel != null) for (int i = 0; i < settingsPanel.getChildCount(); i++) settingsPanel.getChildAt(i).setEnabled(!gate.isRunning());
        if (detSide != null && textDetectionSizeMode != null) detSide.setEnabled(!gate.isRunning()
                && textDetectionSizeMode.getSelectedItemPosition() == TextDetectionSizeMode.MANUAL_LIMIT.ordinal());
        jsonButton.setEnabled(lastResult != null); pngButton.setEnabled(lastResult != null);
    }
    @Override protected void onResume() { super.onResume(); resumed = true; ui.removeCallbacks(ageTicker); ui.post(ageTicker); }
    @Override protected void onStop() { resumed = false; closeCamera(); ui.removeCallbacks(ageTicker); super.onStop(); }
    @Override protected void onDestroy() {
        destroyed = true; gate.stop(); ui.removeCallbacks(ageTicker);
        analyzer.execute(() -> {
            // Earlier coordinator jobs have joined every native branch before this cleanup.
            try {
                if (pipeline != null) try { pipeline.close(); } catch (Exception ignored) {}
                Future<?> closeHands = handWorker.submit(() -> { if (handEngine != null) handEngine.close(); });
                InferenceTasks.drain(closeHands);
            } finally { handWorker.shutdown(); yoloWorker.shutdown(); }
        }); analyzer.shutdown();
        resultImage.show(null, true);matchedRegions.show(null,null,0);
        liveOverlay.clear();if (lastResult != null) lastResult.recycleImages();lastResult=null;super.onDestroy();
    }
    @Override protected void onSaveInstanceState(Bundle state) {
        super.onSaveInstanceState(state);
        state.putString("angle", downwardAngle.getText().toString()); state.putString("knob", knobScore.getText().toString());
        state.putString("text", textScore.getText().toString()); state.putString("detSide", detSide.getText().toString());
        state.putInt("inferenceMode", inferenceMode.getSelectedItemPosition());
        state.putString("textDetectionMode", TextDetectionMode.values()[textDetectionMode.getSelectedItemPosition()].name());
        state.putString("textDetectionSizeMode", TextDetectionSizeMode.values()[textDetectionSizeMode.getSelectedItemPosition()].name());
        state.putInt("yoloChoice", yoloSize.getSelectedItemPosition()); state.putInt("ocrChoice", ocrModel.getSelectedItemPosition());
        state.putString("ocrModelId", selectedModels().ocr.name());
        state.putInt("settingsVersion", 5); state.putInt("searchDirection", searchDirection.getSelectedItemPosition());
        state.putBoolean("classifier", angleClassifier.isChecked()); state.putInt("resolution", resolution.getSelectedItemPosition());
    }
    private TextView label(String text, int size) { TextView view = new TextView(this); view.setText(text); view.setTextSize(size); view.setTextColor(Color.rgb(16, 42, 67)); view.setPadding(0, dp(6), 0, dp(6)); return view; }
    private Button button(String text) { Button view = new Button(this); view.setText(text); return view; }
    private EditText field(String title, String value) {
        settingsPanel.addView(label(title, 13)); EditText input = new EditText(this); input.setSingleLine(true);
        input.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL); input.setText(value); settingsPanel.addView(input); return input;
    }
    private LinearLayout.LayoutParams weight() { return new LinearLayout.LayoutParams(0, -2, 1); }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
    void selectModelsForTest(int side, ModelChoice.Ocr ocr) {
        stopAnalysis(); modelSelectionEnabled = false;
        yoloSize.setSelection(side == 640 ? 0 : 1); ocrModel.setSelection(ocr.ordinal());
        modelSelectionEnabled = true; requestModels(selectedModels());
    }
    ModelChoice loadedChoiceForTest() { return modelsReady ? requestedModels : null; }
    boolean cameraBoundForTest() { return cameraBound; }
    boolean readyForTest() { return modelsReady; }
    FrameResult resultForTest() { return lastResult; }
    void openCameraForTest() { requestCamera(); }
    void startForTest() { startAnalysis(); }
    void stopForTest() { stopAnalysis(); }
    void slowForTest(long milliseconds) { testDelayMs = milliseconds; }
    boolean busyForTest() { return gate.isBusy(); }
    boolean previewStreamingForTest() { return live.getPreviewStreamState().getValue() == PreviewView.StreamState.STREAMING; }
    void setDirectionForTest(SearchDirection direction) { searchDirection.setSelection(direction.ordinal()); }
    void setModeForTest(InferenceMode mode) { inferenceMode.setSelection(mode.ordinal()); }
    void setTextDetectionModeForTest(TextDetectionMode mode) { textDetectionMode.setSelection(mode.ordinal()); }
    TextDetectionMode textDetectionModeForTest() { return TextDetectionMode.values()[textDetectionMode.getSelectedItemPosition()]; }
    void setTextDetectionSizeModeForTest(TextDetectionSizeMode mode) { textDetectionSizeMode.setSelection(mode.ordinal()); }
    TextDetectionSizeMode textDetectionSizeModeForTest() { return TextDetectionSizeMode.values()[textDetectionSizeMode.getSelectedItemPosition()]; }
    boolean manualTextSizeEnabledForTest() { return detSide.isEnabled(); }
    boolean angleEnabledForTest() { return angleClassifier.isChecked(); }
    boolean overlayHasResultForTest() { return liveOverlay.hasResultForTest(); }
}
