package com.example.cabinetmatcher.camera;

import android.graphics.Bitmap;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.example.cabinetmatcher.camera.core.CabinetPipeline;
import com.example.cabinetmatcher.camera.core.MatchReport;
import com.example.cabinetmatcher.camera.core.FrameGate;
import com.example.cabinetmatcher.camera.core.ModelChoice;
import com.example.cabinetmatcher.camera.core.InferenceMode;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class CameraSmokeTest {
    @Test public void rotatedFramesKeepPixelLocationsAndColor() {
        assertTrue(OpenCVLoader.initLocal());
        Mat image = new Mat(2, 3, org.opencv.core.CvType.CV_8UC3);
        try {
            image.put(0, 0, new byte[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});
            CameraFrames.rotate(image, 90); assertEquals(3, image.rows()); assertEquals(2, image.cols());
            assertArrayEquals(new double[]{10,11,12}, image.get(0,0), 0);
            assertArrayEquals(new double[]{1,2,3}, image.get(0,1), 0);
            CameraFrames.rotate(image, 270); assertEquals(2, image.rows()); assertEquals(3, image.cols());
            Bitmap bitmap = CameraFrames.bitmap(image);
            assertEquals(android.graphics.Color.rgb(3,2,1), bitmap.getPixel(0,0)); bitmap.recycle();
        } finally { image.release(); }
    }
    @Test public void fourModelCombinationsMatchBothImagesAndExportMetadata() throws Exception {
        android.content.Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertTrue(OpenCVLoader.initLocal()); File models = ModelAssets.prepare(context);
        org.json.JSONArray results = new org.json.JSONArray();
        for (int side : new int[]{640, 1280}) for (ModelChoice.Ocr ocr : ModelChoice.Ocr.values()) {
            ModelChoice choice = new ModelChoice(side, ocr);
            try (CabinetPipeline pipeline = CabinetPipelines.create(models, choice)) {
                for (int image = 1; image <= 2; image++) {
                    Bitmap bitmap;
                    android.content.res.AssetManager assets = image == 1
                            ? InstrumentationRegistry.getInstrumentation().getContext().getAssets() : context.getAssets();
                    try (java.io.InputStream input = assets.open(image == 1 ? "test_cabinet_1.jpg" : "demo_cabinet.jpg")) {
                        bitmap = android.graphics.BitmapFactory.decodeStream(input);
                    }
                    Mat rgba = new Mat(), bgr = new Mat();
                    try {
                        Utils.bitmapToMat(bitmap, rgba); Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
                        for (int sample = 0; sample < 3; sample++) {
                            CabinetPipeline.Settings settings = new CabinetPipeline.Settings(); settings.models = choice;
                            FrameGate gate = new FrameGate(); gate.start(); FrameGate.Ticket ticket = gate.tryBegin(System.nanoTime());
                            MatchReport report = pipeline.run(bgr, settings, message -> {});
                            report.timing.modelsReused = true;
                            assertFalse("New knob model must detect knobs", report.knobs.isEmpty());
                            assertFalse("OCR detection must produce regions", report.texts.isEmpty());
                            if (image == 2) assertTrue("Demo must have geometric matches", report.matches.size() > 0);
                            assertEquals(report.matches.size(), report.timing.recognitionModelCalls);
                            for (com.example.cabinetmatcher.camera.core.Detection text : report.texts)
                                assertNull("Selected regions must decode without errors", text.recognitionError);
                            assertTrue(report.timing.millis("yolo.inference") > 0);
                            assertTrue(report.timing.millis("ocr_detection.inference") > 0);
                            Bitmap annotation = Annotations.draw(bitmap, report);
                            MainActivity.FrameResult result = new MainActivity.FrameResult(ticket, report, annotation, settings,
                                    123, System.nanoTime(), bgr.cols(), bgr.rows(), 0);
                            result.modelLoadMs = pipeline.initializationTiming.millis("model_load.total");
                            result.firstInferenceForModels = image == 1 && sample == 0;
                            JSONObject json = new JSONObject(result.json());
                            assertEquals(side, json.getJSONObject("models").getInt("yolo_input_size"));
                            assertEquals(ocr.name(), json.getJSONObject("models").getString("ocr_model"));
                            if (ocr.usesMlKit()) assertTrue(json.getJSONObject("models").isNull("dictionary_classes"));
                            else assertEquals(ocr.classes, json.getJSONObject("models").getInt("dictionary_classes"));
                            assertEquals("live_preview_and_processed_frame", json.getJSONObject("video").getString("result_overlay_target"));
                            assertEquals(1, json.getJSONObject("video").getInt("max_concurrent_inferences"));
                            assertEquals("KEEP_ONLY_LATEST", json.getJSONObject("video").getString("backpressure_strategy"));
                            assertEquals(bgr.cols(), json.getJSONObject("image_size").getInt("width"));
                            json.put("benchmark", new JSONObject().put("image", image + ".jpg").put("sample", sample)
                                    .put("warm_sample", sample > 0).put("scope", "fixed image pipeline, not camera FPS"));
                            results.put(json);
                            if (sample == 2) try (FileOutputStream output = new FileOutputStream(new File(context.getFilesDir(),
                                    "matrix-" + side + "-" + ocr.name() + "-" + image + ".png"))) {
                                assertTrue(annotation.compress(Bitmap.CompressFormat.PNG,100,output));
                            }
                            annotation.recycle(); gate.finish(ticket);
                        }
                    } finally { bitmap.recycle(); rgba.release(); bgr.release(); }
                }
            }
        }
        try (FileOutputStream output = new FileOutputStream(new File(context.getFilesDir(), "model-matrix-results.json"))) {
            output.write(results.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }
    @Test public void realCameraPreviewContinuesAcrossTwoAnalysesAndPause() throws Exception {
        // Grant CAMERA through adb before this test. Emulator uses its synthetic rear camera.
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            AtomicReference<MainActivity> ref = new AtomicReference<>(); scenario.onActivity(ref::set);
            MainActivity activity = ref.get(); long deadline = System.nanoTime() + 120_000_000_000L;
            while (!activity.readyForTest() && System.nanoTime() < deadline) Thread.sleep(100);
            assertTrue("Models must load", activity.readyForTest());
            scenario.onActivity(a -> a.selectModelsForTest(640, ModelChoice.Ocr.ML_KIT_CHINESE));
            waitUntil(activity::readyForTest, 120000);
            assertEquals(ModelChoice.Ocr.ML_KIT_CHINESE, activity.loadedChoiceForTest().ocr);
            scenario.onActivity(MainActivity::openCameraForTest);
            waitUntil(() -> { AtomicReference<Boolean> bound = new AtomicReference<>(); scenario.onActivity(a -> bound.set(a.cameraBoundForTest())); return bound.get(); }, 30000);
            // Binding completes before the first preview frame is drawn. Wait for that startup separately.
            waitUntil(() -> { AtomicReference<Boolean> streaming = new AtomicReference<>(); scenario.onActivity(a -> streaming.set(a.previewStreamingForTest())); return streaming.get(); }, 30000);
            scenario.onActivity(a -> { a.setModeForTest(InferenceMode.PARALLEL); a.slowForTest(2000); a.startForTest(); });
            waitUntil(activity::busyForTest, 10000);
            Thread.sleep(500);
            AtomicReference<Boolean> streaming = new AtomicReference<>(); scenario.onActivity(a -> streaming.set(a.previewStreamingForTest()));
            assertTrue("Preview must stream while slow inference holds the analysis frame", streaming.get());
            waitUntil(() -> activity.resultForTest() != null, 120000);
            MainActivity.FrameResult first = activity.resultForTest();
            assertTrue("Each admitted frame must finish hand detection before publishing", first.handDetectionCompleted);
            assertTrue(first.report.timing.millis("hands.inference") > 0);
            assertTrue(first.settings.angleClassifier);
            scenario.onActivity(a -> assertTrue("Text orientation must default on", a.angleEnabledForTest()));
            scenario.onActivity(a -> assertTrue("Completed results must also be on the live preview", a.overlayHasResultForTest()));
            waitUntil(() -> activity.resultForTest().ticket.sequence > first.ticket.sequence, 120000);
            MainActivity.FrameResult second = activity.resultForTest();
            assertTrue(second.handDetectionCompleted);
            JSONObject sameFrame = new JSONObject(second.json()).getJSONObject("hand_landmarks");
            assertEquals(second.ticket.sequence, sameFrame.getLong("frame_id"));
            assertEquals(second.cameraTimestampNs, sameFrame.getLong("camera_timestamp_ns"));
            assertTrue(sameFrame.getBoolean("same_frame_as_cabinet"));
            assertTrue("Next frame must be fresh, not the next queued 30fps frame",
                    second.cameraTimestampNs - first.cameraTimestampNs > 1_000_000_000L);
            assertTrue(second.finishedNs - second.ticket.receivedNs >= 2_000_000_000L);
            assertTrue(second.report.width > 0);
            scenario.onActivity(a -> {
                a.stopForTest();
                try {
                    android.content.Context context = a;
                    try (FileOutputStream output = new FileOutputStream(new File(context.getFilesDir(), "camera-live-result.json"))) {
                        output.write(a.resultForTest().json().getBytes(StandardCharsets.UTF_8));
                    }
                    try (FileOutputStream output = new FileOutputStream(new File(context.getFilesDir(), "camera-live-frame.png"))) {
                        assertTrue(a.resultForTest().annotated.compress(Bitmap.CompressFormat.PNG,100,output));
                    }
                } catch (Exception error) { throw new AssertionError(error); }
            });
            // Pause during another analysis, change both selectors, then resume. Old results must stay labelled old.
            scenario.onActivity(a -> { a.slowForTest(2000); a.startForTest(); });
            waitUntil(activity::busyForTest, 10000);
            ModelChoice next = new ModelChoice(1280, ModelChoice.Ocr.DBNET_CRNN);
            scenario.onActivity(a -> a.selectModelsForTest(next.yoloSide, next.ocr));
            scenario.onActivity(a -> assertFalse("Switching must clear old-model preview overlays", a.overlayHasResultForTest()));
            assertFalse("Changing models must disable start until loading is complete", activity.readyForTest());
            long oldSequence = activity.resultForTest().ticket.sequence;
            waitUntil(activity::readyForTest, 120000);
            assertEquals("Cancelled old-model frame must not publish", oldSequence, activity.resultForTest().ticket.sequence);
            scenario.onActivity(a -> { assertEquals(next, a.loadedChoiceForTest()); a.slowForTest(0); a.startForTest(); });
            waitUntil(() -> activity.resultForTest().settings.models.equals(next), 120000);
            scenario.onActivity(MainActivity::stopForTest);
            assertEquals(next.yoloSide, new JSONObject(activity.resultForTest().json()).getJSONObject("models").getInt("yolo_input_size"));
            Bitmap screen = InstrumentationRegistry.getInstrumentation().getUiAutomation().takeScreenshot();
            try (FileOutputStream output = new FileOutputStream(new File(activity.getFilesDir(), "camera-screen.png"))) {
                assertTrue(screen.compress(Bitmap.CompressFormat.PNG,100,output));
            } finally { screen.recycle(); }
            long stopped = activity.resultForTest().ticket.sequence; Thread.sleep(2500);
            assertEquals("Paused/stale results must not replace the displayed frame", stopped, activity.resultForTest().ticket.sequence);
            scenario.moveToState(androidx.lifecycle.Lifecycle.State.CREATED);
            AtomicReference<Boolean> bound = new AtomicReference<>(); scenario.onActivity(a -> {
                bound.set(a.cameraBoundForTest()); assertFalse("Closing camera must clear overlay", a.overlayHasResultForTest());
            }); assertFalse(bound.get());
            scenario.moveToState(androidx.lifecycle.Lifecycle.State.RESUMED);
        }
    }
    private interface Condition { boolean get(); }
    private static void waitUntil(Condition condition, long milliseconds) throws Exception {
        long deadline = System.nanoTime() + milliseconds * 1_000_000L;
        while (!condition.get() && System.nanoTime() < deadline) Thread.sleep(100);
        assertTrue("Timed out", condition.get());
    }
}
