package com.example.cabinetmatcher.camera;

import android.content.Context;
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
public class OcrOptionsTest {
    @Test public void everyBundledOcrRunsRealImageAndExportsItsOwnIdentity() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertTrue(OpenCVLoader.initLocal());
        File models = ModelAssets.prepare(context);
        Bitmap bitmap;
        try (InputStream input = context.getAssets().open("demo_cabinet.jpg")) {
            bitmap = BitmapFactory.decodeStream(input);
        }
        Mat rgba = new Mat(), bgr = new Mat();
        JSONArray results = new JSONArray();
        try {
            Utils.bitmapToMat(bitmap, rgba);
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
            for (ModelChoice.Ocr ocr : ModelChoice.Ocr.values()) {
                ModelChoice choice = new ModelChoice(640, ocr);
                try (CabinetPipeline pipeline = CabinetPipelines.create(models, choice)) {
                    CabinetPipeline.Settings settings = new CabinetPipeline.Settings();
                    settings.models = choice;
                    MatchReport report = pipeline.run(bgr, settings, message -> {});
                    assertFalse(ocr.label + " must detect text", report.texts.isEmpty());
                    assertFalse(ocr.label + " must produce matches", report.matches.isEmpty());
                    assertEquals(report.matches.size(), report.timing.recognitionModelCalls);
                    boolean decoded = false;
                    for (Detection region : report.texts) {
                        assertNull(ocr.label + ": " + region.recognitionError, region.recognitionError);
                        if (region.text != null && !region.text.trim().isEmpty()) decoded = true;
                    }
                    assertTrue(ocr.label + " must decode at least one matched region", decoded);
                    FrameGate gate = new FrameGate(); gate.start();
                    FrameGate.Ticket ticket = gate.tryBegin(System.nanoTime());
                    MainActivity.FrameResult frame = new MainActivity.FrameResult(ticket, report, null, settings,
                            123, System.nanoTime(), bgr.cols(), bgr.rows(), 0);
                    frame.modelLoadMs = pipeline.initializationTiming.millis("model_load.total");
                    JSONObject json = new JSONObject(frame.json());
                    JSONObject identity = json.getJSONObject("models");
                    assertEquals(ocr.name(), identity.getString("ocr_model"));
                    if (ocr.usesMlKit()) {
                        assertTrue(identity.isNull("ocr_recognition_file"));
                        assertTrue(identity.isNull("dictionary_file"));
                        assertTrue(identity.isNull("dictionary_classes"));
                        assertTrue(json.getJSONObject("ocr_engine").getBoolean("whole_image_sdk_reads_text"));
                    } else {
                        assertEquals(ocr.recognitionFile, identity.getString("ocr_recognition_file"));
                        assertEquals(ocr.dictionaryFile, identity.getString("dictionary_file"));
                        assertEquals(ocr.classes, identity.getInt("dictionary_classes"));
                        assertEquals(ocr.recognitionHeight(), identity.getInt("recognition_input_height"));
                    }
                    results.put(json); gate.finish(ticket);
                    // Save after each model so a slow server run still leaves diagnostic evidence.
                    try (FileOutputStream output = new FileOutputStream(new File(context.getFilesDir(), "ocr-options-results.json"))) {
                        output.write(results.toString(2).getBytes(StandardCharsets.UTF_8));
                    }
                }
            }
        } finally { bitmap.recycle(); rgba.release(); bgr.release(); }
    }

    @Test public void modelMenuSelectsAllOptionsAndRestoresSelectionAfterRecreation() throws Exception {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            for (ModelChoice.Ocr ocr : ModelChoice.Ocr.values()) {
                ModelChoice choice = new ModelChoice(640, ocr);
                scenario.onActivity(activity -> activity.selectModelsForTest(640, ocr));
                waitForChoice(scenario, choice);
            }
            scenario.recreate();
            waitForChoice(scenario, new ModelChoice(640, ModelChoice.Ocr.DBNET_CRNN));
        }
        // Launch without saved instance state to verify SharedPreferences, too.
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            waitForChoice(scenario, new ModelChoice(640, ModelChoice.Ocr.DBNET_CRNN));
        }
    }

    @Test public void newEnginesAgreeBetweenSerialAndParallelAndKeepConfidenceFiltering() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertTrue(OpenCVLoader.initLocal());
        Bitmap bitmap;
        try (InputStream input = context.getAssets().open("demo_cabinet.jpg")) {
            bitmap = BitmapFactory.decodeStream(input);
        }
        Mat rgba = new Mat(), bgr = new Mat();
        java.util.concurrent.ExecutorService worker = java.util.concurrent.Executors.newSingleThreadExecutor();
        try {
            Utils.bitmapToMat(bitmap, rgba); Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
            for (ModelChoice.Ocr ocr : new ModelChoice.Ocr[]{ModelChoice.Ocr.ML_KIT_CHINESE, ModelChoice.Ocr.DBNET_CRNN}) {
                ModelChoice choice = new ModelChoice(640, ocr);
                try (CabinetPipeline pipeline = CabinetPipelines.create(ModelAssets.prepare(context), choice)) {
                    CabinetPipeline.Settings settings = new CabinetPipeline.Settings(); settings.models = choice;
                    MatchReport serial = pipeline.run(bgr, settings, message -> {});
                    settings.inferenceMode = InferenceMode.PARALLEL;
                    MatchReport parallel = pipeline.run(bgr, settings, message -> {}, worker);
                    assertEquals(serial.texts.size(), parallel.texts.size());
                    assertEquals(serial.matches.size(), parallel.matches.size());
                    for (int i = 0; i < serial.texts.size(); i++) {
                        Detection a = serial.texts.get(i), b = parallel.texts.get(i);
                        assertArrayEquals(a.box, b.box, .001); assertEquals(a.text, b.text);
                        assertEquals(a.recognitionStatus, b.recognitionStatus);
                        assertEquals(a.confidence, b.confidence, .0001);
                        if (a.recognitionAttempted && a.text != null && !a.text.isEmpty()) {
                            double originalScore = a.confidence;
                            RecognitionSelection.record(a, a.text, a.confidence, Math.min(1, a.confidence + .001));
                            if (originalScore < 1) assertEquals("low_confidence", a.recognitionStatus);
                        }
                    }
                }
            }
        } finally { worker.shutdown(); bitmap.recycle(); rgba.release(); bgr.release(); }
    }

    private static void waitForChoice(ActivityScenario<MainActivity> scenario, ModelChoice choice) throws Exception {
        long deadline = System.nanoTime() + 180_000_000_000L;
        java.util.concurrent.atomic.AtomicReference<ModelChoice> actual = new java.util.concurrent.atomic.AtomicReference<>();
        do {
            scenario.onActivity(activity -> actual.set(activity.loadedChoiceForTest()));
            if (choice.equals(actual.get())) return;
            Thread.sleep(100);
        } while (System.nanoTime() < deadline);
        fail("Model menu did not load " + choice.label());
    }
}
