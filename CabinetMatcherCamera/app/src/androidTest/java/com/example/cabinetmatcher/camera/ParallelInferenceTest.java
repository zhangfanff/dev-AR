package com.example.cabinetmatcher.camera;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.example.cabinetmatcher.camera.core.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class ParallelInferenceTest {
    @Test public void realModelsPreserveDetectionsMatchesAndTextForAllFourModelCombinations() throws Exception {
        android.content.Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertTrue(OpenCVLoader.initLocal());
        Bitmap input;
        try (InputStream stream = context.getAssets().open("demo_cabinet.jpg")) { input = BitmapFactory.decodeStream(stream); }
        Mat rgba = new Mat(), bgr = new Mat();
        ExecutorService yolo = Executors.newSingleThreadExecutor(), hands = Executors.newSingleThreadExecutor();
        JSONArray measurements = new JSONArray();
        try {
            Utils.bitmapToMat(input, rgba); Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
            for (int side : new int[]{640, 1280}) for (ModelChoice.Ocr ocr : ModelChoice.Ocr.values()) {
                ModelChoice choice = new ModelChoice(side, ocr);
                try (CabinetPipeline pipeline = CabinetPipelines.create(ModelAssets.prepare(context), choice)) {
                    MatchReport serial = null;
                    for (InferenceMode mode : InferenceMode.values()) {
                        CabinetPipeline.Settings settings = new CabinetPipeline.Settings(); settings.models = choice; settings.inferenceMode = mode;
                        // Construction/detect/close stay on the exact same persistent thread.
                        HandEngine hand = InferenceTasks.await(hands.submit(() -> new HandEngine(context)));
                        try {
                            for (int sample = 0; sample < 2; sample++) {
                                Future<float[][]> handTask = null; MatchReport report;
                                long started = System.nanoTime();
                                final long handTimestamp = 1000 + sample * 1000;
                                try {
                                    if (mode == InferenceMode.PARALLEL)
                                        handTask = hands.submit(() -> detectHands(hand, bgr, handTimestamp));
                                    report = pipeline.run(bgr, settings, message -> {}, yolo);
                                    if (handTask == null) handTask = hands.submit(() -> detectHands(hand, bgr, handTimestamp));
                                    assertTrue(InferenceTasks.await(handTask).length <= 2);
                                } finally { InferenceTasks.drain(handTask); }
                                assertFalse(report.knobs.isEmpty()); assertFalse(report.texts.isEmpty()); assertFalse(report.matches.isEmpty());
                                assertEquals(report.matches.size(), report.timing.recognitionModelCalls);
                                for (Detection text : report.texts) assertNull(text.recognitionError);
                                assertTrue(report.timing.nanos("detection.wall") > 0);
                                if (mode == InferenceMode.SERIAL) serial = report;
                                else assertSameDetectionsAndText(serial, report);
                                measurements.put(new JSONObject().put("yolo_input", side).put("ocr", ocr.name()).put("mode", mode.name()).put("sample", sample)
                                        .put("warmup", sample == 0).put("total_compute_wall_ms", (System.nanoTime()-started)/1e6)
                                        .put("cabinet_wall_ms", report.timing.millis("pipeline.total"))
                                        .put("detection_join_wall_ms", report.timing.millis("detection.wall"))
                                        .put("yolo_ms", report.timing.millis("yolo.total"))
                                        .put("ocr_detection_ms", report.timing.millis("ocr_detection.total"))
                                        .put("matches", report.matches.size()).put("recognition_ms", report.timing.millis("text_loop.total")));
                            }
                        } finally { InferenceTasks.drain(hands.submit(hand::close)); }
                    }
                }
            }
            JSONObject output = new JSONObject().put("device", "Android x86_64 local emulator; NOT phone performance")
                    .put("yolo_inputs", new JSONArray(new int[]{640,1280})).put("ocr_detection_limit", 1600)
                    .put("note", "Functional equivalence check; one measured sample after warmup per mode, order/thermal effects uncontrolled; not a speedup benchmark")
                    .put("measurements", measurements);
            try (FileOutputStream stream = new FileOutputStream(new File(context.getFilesDir(), "parallel-model-comparison.json"))) {
                stream.write(output.toString(2).getBytes(StandardCharsets.UTF_8));
            }
        } finally {
            yolo.shutdown(); hands.shutdown(); rgba.release(); bgr.release(); if (!input.isRecycled()) input.recycle();
        }
    }
    private static float[][] detectHands(HandEngine hand, Mat bgr, long timestamp) {
        Bitmap bitmap = CameraFrames.bitmap(bgr);
        try { return hand.detect(bitmap, timestamp); }
        finally { if (!bitmap.isRecycled()) bitmap.recycle(); }
    }
    private static void assertSameDetectionsAndText(MatchReport serial, MatchReport parallel) {
        assertEquals(serial.knobs.size(), parallel.knobs.size()); assertEquals(serial.texts.size(), parallel.texts.size());
        assertEquals(serial.matches.size(), parallel.matches.size());
        for (int i = 0; i < serial.knobs.size(); i++) {
            assertArrayEquals(serial.knobs.get(i).box, parallel.knobs.get(i).box, 1e-4);
            assertEquals(serial.knobs.get(i).confidence, parallel.knobs.get(i).confidence, 1e-5);
        }
        for (int i = 0; i < serial.texts.size(); i++) {
            Detection expected = serial.texts.get(i), actual = parallel.texts.get(i);
            assertArrayEquals(expected.box, actual.box, 1e-4); assertEquals(expected.text, actual.text);
            assertEquals(expected.recognitionStatus, actual.recognitionStatus);
            assertEquals(expected.confidence, actual.confidence, 1e-5);
        }
        for (int i = 0; i < serial.matches.size(); i++) {
            assertEquals(serial.matches.get(i).knob.id, parallel.matches.get(i).knob.id);
            assertEquals(serial.matches.get(i).text.id, parallel.matches.get(i).text.id);
        }
    }

    @Test public void cameraSwitchesModesWithoutPublishingStaleOrMixedFrameResults() throws Exception {
        AtomicReference<MainActivity> ref = new AtomicReference<>();
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            scenario.onActivity(ref::set); MainActivity activity = ref.get();
            waitUntil(activity::readyForTest, 120000);
            scenario.onActivity(a -> { a.selectModelsForTest(640, ModelChoice.Ocr.V6_TINY); a.setModeForTest(InferenceMode.SERIAL); a.openCameraForTest(); });
            waitUntil(() -> activity.readyForTest() && activity.cameraBoundForTest(), 120000);
            scenario.onActivity(MainActivity::startForTest);
            waitUntil(() -> activity.resultForTest() != null, 120000);
            MainActivity.FrameResult baseline = activity.resultForTest(); assertEquals(InferenceMode.SERIAL, baseline.settings.inferenceMode);
            assertTrue(baseline.handDetectionCompleted);
            scenario.onActivity(a -> { a.stopForTest(); a.slowForTest(1000); a.setModeForTest(InferenceMode.PARALLEL); a.startForTest(); });
            waitUntil(() -> activity.resultForTest().settings.inferenceMode == InferenceMode.PARALLEL, 120000);
            MainActivity.FrameResult first = activity.resultForTest();
            waitUntil(() -> activity.resultForTest().ticket.sequence > first.ticket.sequence, 120000);
            MainActivity.FrameResult second = activity.resultForTest();
            assertEquals(InferenceMode.PARALLEL, second.settings.inferenceMode); assertTrue(second.handDetectionCompleted);
            JSONObject json = new JSONObject(second.json());
            assertEquals(1, json.getJSONObject("video").getInt("max_concurrent_frames"));
            assertEquals(3, json.getJSONObject("execution").getInt("max_concurrent_model_tasks"));
            assertTrue(json.getJSONObject("execution").getBoolean("all_branches_joined_before_publish"));
            assertEquals(second.ticket.sequence, json.getJSONObject("hand_landmarks").getLong("frame_id"));
            assertEquals(second.cameraTimestampNs, json.getJSONObject("hand_landmarks").getLong("camera_timestamp_ns"));
            assertTrue(second.cameraTimestampNs-first.cameraTimestampNs > 500_000_000L);
            scenario.onActivity(a -> { a.stopForTest(); a.slowForTest(1500); a.startForTest(); });
            waitUntil(activity::busyForTest, 10000);
            scenario.onActivity(MainActivity::stopForTest);
            long stopped = activity.resultForTest().ticket.sequence;
            waitUntil(() -> !activity.busyForTest(), 120000);
            assertEquals(stopped, activity.resultForTest().ticket.sequence);
            scenario.onActivity(a -> { a.setModeForTest(InferenceMode.SERIAL); a.slowForTest(0); a.startForTest(); });
            waitUntil(() -> activity.resultForTest().ticket.sequence > stopped, 120000);
            assertEquals(InferenceMode.SERIAL, activity.resultForTest().settings.inferenceMode);
            scenario.onActivity(MainActivity::stopForTest);
            try (FileOutputStream stream = new FileOutputStream(new File(activity.getFilesDir(), "parallel-camera-result.json"))) {
                stream.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
            }
        }
    }
    private interface Condition { boolean get(); }
    private static void waitUntil(Condition condition, long milliseconds) throws Exception {
        long deadline = System.nanoTime()+milliseconds*1_000_000L;
        while (!condition.get() && System.nanoTime()<deadline) Thread.sleep(100);
        assertTrue("Timed out", condition.get());
    }
}
