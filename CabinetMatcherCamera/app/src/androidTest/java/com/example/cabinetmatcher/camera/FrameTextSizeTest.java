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
public class FrameTextSizeTest {
    @Test public void paddleAndMlKitFollowActualFrameInSingleAndFourDirections() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertTrue(OpenCVLoader.initLocal());
        Bitmap bitmap;
        try (InputStream stream = context.getAssets().open("demo_cabinet.jpg")) { bitmap = BitmapFactory.decodeStream(stream); }
        Mat rgba = new Mat(), bgr = new Mat();
        JSONArray results = new JSONArray();
        try {
            Utils.bitmapToMat(bitmap,rgba); Imgproc.cvtColor(rgba,bgr,Imgproc.COLOR_RGBA2BGR);
            for (ModelChoice.Ocr ocr : new ModelChoice.Ocr[]{ModelChoice.Ocr.V4,ModelChoice.Ocr.ML_KIT_CHINESE}) {
                ModelChoice choice = new ModelChoice(640,ocr);
                try (CabinetPipeline pipeline = CabinetPipelines.create(ModelAssets.prepare(context),choice)) {
                    for (TextDetectionMode directions : TextDetectionMode.values()) {
                        CabinetPipeline.Settings settings = new CabinetPipeline.Settings(); settings.models = choice;
                        settings.detSide = 640; // An intentionally small manual limit must be ignored in follow mode.
                        settings.textDetectionSizeMode = TextDetectionSizeMode.FOLLOW_FRAME;
                        settings.textDetectionMode = directions;
                        MatchReport report = pipeline.run(bgr,settings,message -> {});
                        assertFalse(report.matches.isEmpty());
                        assertEquals(report.matches.size(),report.timing.recognitionModelCalls);
                        assertEquals(directions.rotations().length,report.timing.detectionPasses.size());
                        for (Detection region : report.texts) assertNull(region.recognitionError);
                        for (TimingReport.DetectionPass pass : report.timing.detectionPasses) {
                            boolean swapped = pass.rotationDegrees == 90 || pass.rotationDegrees == 270;
                            int width = swapped ? bgr.rows() : bgr.cols(), height = swapped ? bgr.cols() : bgr.rows();
                            assertEquals(width,pass.timing.detectionSourceWidth);
                            assertEquals(height,pass.timing.detectionSourceHeight);
                            int[] expected = TextDetectionSizeMode.inputSize(width,height,Math.max(width,height),!ocr.usesMlKit());
                            assertEquals(expected[0],pass.timing.detectionInputWidth);
                            assertEquals(expected[1],pass.timing.detectionInputHeight);
                            assertTrue(Math.max(expected[0],expected[1]) > settings.detSide);
                        }
                        FrameGate gate = new FrameGate(); gate.start();
                        FrameGate.Ticket ticket = gate.tryBegin(System.nanoTime());
                        JSONObject json = new JSONObject(new MainActivity.FrameResult(ticket,report,null,settings,
                                123,System.nanoTime(),bgr.cols(),bgr.rows(),0).json());
                        assertEquals("FOLLOW_FRAME",json.getJSONObject("parameters").getString("text_detection_size_mode"));
                        assertTrue(json.getJSONObject("parameters").isNull("ocr_det_side"));
                        JSONArray passes = json.getJSONObject("timings").getJSONArray("text_detection_passes");
                        for (int i = 0; i < passes.length(); i++) {
                            JSONObject input = passes.getJSONObject(i).getJSONObject("detector_input_size");
                            assertEquals(report.timing.detectionPasses.get(i).timing.detectionInputWidth,input.getInt("width"));
                            assertEquals(report.timing.detectionPasses.get(i).timing.detectionInputHeight,input.getInt("height"));
                        }
                        assertTrue(TimingDisplay.details(report.timing).contains("跟随相机分析帧"));
                        results.put(json); gate.finish(ticket);
                        try (FileOutputStream out = new FileOutputStream(new File(context.getFilesDir(),"frame-text-size-results.json"))) {
                            out.write(results.toString(2).getBytes(StandardCharsets.UTF_8));
                        }
                    }
                }
            }
        } finally { bitmap.recycle(); rgba.release(); bgr.release(); }
    }

    @Test public void sizeChoicePersistsAndDisablesManualFieldOnlyInFollowMode() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        android.content.SharedPreferences prefs = context.getSharedPreferences("MainActivity",Context.MODE_PRIVATE);
        String previous = prefs.getString("textDetectionSizeMode",null);
        prefs.edit().remove("textDetectionSizeMode").commit();
        try {
            try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
                scenario.onActivity(activity -> {
                    assertEquals(TextDetectionSizeMode.MANUAL_LIMIT,activity.textDetectionSizeModeForTest());
                    assertTrue(activity.manualTextSizeEnabledForTest());
                    activity.setTextDetectionSizeModeForTest(TextDetectionSizeMode.FOLLOW_FRAME);
                });
                InstrumentationRegistry.getInstrumentation().waitForIdleSync();
                scenario.onActivity(activity -> assertFalse(activity.manualTextSizeEnabledForTest()));
                scenario.recreate();
                scenario.onActivity(activity -> {
                    assertEquals(TextDetectionSizeMode.FOLLOW_FRAME,activity.textDetectionSizeModeForTest());
                    assertFalse(activity.manualTextSizeEnabledForTest());
                });
            }
            try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
                scenario.onActivity(activity -> {
                    assertEquals(TextDetectionSizeMode.FOLLOW_FRAME,activity.textDetectionSizeModeForTest());
                    activity.setTextDetectionSizeModeForTest(TextDetectionSizeMode.MANUAL_LIMIT);
                });
                InstrumentationRegistry.getInstrumentation().waitForIdleSync();
                scenario.onActivity(activity -> assertTrue(activity.manualTextSizeEnabledForTest()));
            }
            try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
                scenario.onActivity(activity -> assertEquals(TextDetectionSizeMode.MANUAL_LIMIT,activity.textDetectionSizeModeForTest()));
            }
        } finally {
            if (previous == null) prefs.edit().remove("textDetectionSizeMode").commit();
            else prefs.edit().putString("textDetectionSizeMode",previous).commit();
        }
    }
}
