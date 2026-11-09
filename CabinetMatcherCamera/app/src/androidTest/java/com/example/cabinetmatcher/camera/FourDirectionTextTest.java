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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class FourDirectionTextTest {
    private static double[][] box(double x, double y, double w, double h) {
        return new double[][]{{x,y},{x+w,y},{x+w,y+h},{x,y+h}};
    }
    private static double[][] forward(double[][] polygon, int degrees, int w, int h) {
        double[][] result = new double[polygon.length][2];
        for (int i = 0; i < polygon.length; i++) {
            double x = polygon[i][0], y = polygon[i][1];
            if (degrees == 0) result[i] = new double[]{x,y};
            if (degrees == 90) result[i] = new double[]{h-1-y,x};
            if (degrees == 180) result[i] = new double[]{w-1-x,h-1-y};
            if (degrees == 270) result[i] = new double[]{y,w-1-x};
        }
        return result;
    }
    @Test public void rotatesPixelsRestoresBoxesRecoversMissAndKeepsSingleModeUnchanged() throws Exception {
        assertTrue(OpenCVLoader.initLocal());
        Mat image = new Mat(50,80,CvType.CV_8UC3,Scalar.all(0));
        image.put(3,5,new byte[]{11,22,33});
        Mat original = image.clone();
        try {
            AtomicInteger calls = new AtomicInteger();
            TimingReport timing = new TimingReport();
            List<double[][]> result = DirectionalTextDetector.detect(image,TextDetectionMode.FOUR_DIRECTIONS,(input, cost) -> {
                int index = calls.getAndIncrement(), degrees = index*90;
                assertEquals(index%2 == 0 ? 80 : 50,input.cols());
                assertEquals(index%2 == 0 ? 50 : 80,input.rows());
                double[][] marker = forward(new double[][]{{5,3}},degrees,80,50);
                assertArrayEquals(new double[]{11,22,33},input.get((int)marker[0][1],(int)marker[0][0]),0);
                List<double[][]> found = new ArrayList<>();
                found.add(forward(box(10,10,20,8),degrees,80,50));
                if (degrees == 90) found.add(forward(box(45,30,20,8),degrees,80,50));
                cost.add("ocr_detection.inference",100);
                return found;
            },timing,message -> {});
            assertEquals(4,calls.get()); assertEquals(5,timing.rawTextRegionCount); assertEquals(2,result.size());
            assertEquals(4,timing.detectionPasses.size()); assertEquals(400,timing.nanos("ocr_detection.inference"));
            assertEquals(1,TextRegionMerger.iou(box(45,30,20,8),result.get(1)),1e-9);
            assertEquals(0,Core.norm(image,original),0);
            List<double[][]> unchanged = Collections.singletonList(box(10,10,20,8));
            calls.set(0); timing = new TimingReport();
            assertSame(unchanged,DirectionalTextDetector.detect(image,TextDetectionMode.SINGLE,(input, cost) -> {
                calls.incrementAndGet(); assertSame(image,input); return unchanged;
            },timing,message -> {}));
            assertEquals(1,calls.get()); assertEquals(1,timing.detectionPasses.size());
            assertEquals(1,timing.rawTextRegionCount);
        } finally { original.release(); image.release(); }
    }

    @Test public void failedPassPropagatesWithoutContinuingOrModifyingSource() throws Exception {
        assertTrue(OpenCVLoader.initLocal());
        Mat image = new Mat(20,40,CvType.CV_8UC3,Scalar.all(37));
        AtomicInteger calls = new AtomicInteger();
        try {
            try {
                DirectionalTextDetector.detect(image,TextDetectionMode.FOUR_DIRECTIONS,(input,timing) -> {
                    if (calls.incrementAndGet() == 2) throw new IllegalStateException("test detector failure");
                    return Collections.emptyList();
                },new TimingReport(),message -> {});
                fail("Expected detector failure");
            } catch (IllegalStateException expected) { assertEquals("test detector failure",expected.getMessage()); }
            assertEquals(2,calls.get()); assertArrayEquals(new double[]{37,37,37},image.get(0,0),0);
        } finally { image.release(); }
    }

    @Test public void realPaddleAndMlKitRunBothModesAndExportActualPassCounts() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertTrue(OpenCVLoader.initLocal());
        Bitmap bitmap;
        try (InputStream stream = context.getAssets().open("demo_cabinet.jpg")) { bitmap = BitmapFactory.decodeStream(stream); }
        Mat rgba = new Mat(), bgr = new Mat();
        ExecutorService worker = Executors.newSingleThreadExecutor();
        JSONArray outputs = new JSONArray();
        try {
            Utils.bitmapToMat(bitmap,rgba); Imgproc.cvtColor(rgba,bgr,Imgproc.COLOR_RGBA2BGR);
            for (ModelChoice.Ocr ocr : new ModelChoice.Ocr[]{ModelChoice.Ocr.V4,ModelChoice.Ocr.ML_KIT_CHINESE}) {
                ModelChoice choice = new ModelChoice(640,ocr);
                try (CabinetPipeline pipeline = CabinetPipelines.create(ModelAssets.prepare(context),choice)) {
                    MatchReport fourSerial = null;
                    for (int run = 0; run < 3; run++) {
                        CabinetPipeline.Settings settings = new CabinetPipeline.Settings(); settings.models = choice;
                        settings.textDetectionMode = run == 0 ? TextDetectionMode.SINGLE : TextDetectionMode.FOUR_DIRECTIONS;
                        settings.inferenceMode = run == 2 ? InferenceMode.PARALLEL : InferenceMode.SERIAL;
                        MatchReport report = pipeline.run(bgr,settings,message -> {},worker);
                        assertEquals(run == 0 ? 1 : 4,report.timing.detectionPasses.size());
                        assertFalse(report.matches.isEmpty());
                        assertEquals(report.matches.size(),report.timing.recognitionModelCalls);
                        for (Detection text : report.texts) {
                            assertNull(text.recognitionError);
                            for (double[] point : text.polygon) {
                                assertTrue(point[0] >= 0 && point[0] < bgr.cols());
                                assertTrue(point[1] >= 0 && point[1] < bgr.rows());
                            }
                        }
                        int raw = 0;
                        for (TimingReport.DetectionPass pass : report.timing.detectionPasses) raw += pass.regionCount;
                        assertEquals(raw,report.timing.rawTextRegionCount);
                        if (run != 0) assertTrue(report.texts.size() < raw);
                        if (run == 1) fourSerial = report;
                        if (run == 2) {
                            assertEquals(fourSerial.texts.size(),report.texts.size());
                            assertEquals(fourSerial.matches.size(),report.matches.size());
                            for (int i = 0; i < report.texts.size(); i++) {
                                assertArrayEquals(fourSerial.texts.get(i).box,report.texts.get(i).box,.001);
                                assertEquals(fourSerial.texts.get(i).text,report.texts.get(i).text);
                            }
                        }
                        FrameGate gate = new FrameGate(); gate.start();
                        FrameGate.Ticket ticket = gate.tryBegin(System.nanoTime());
                        JSONObject json = new JSONObject(new MainActivity.FrameResult(ticket,report,null,settings,
                                123,System.nanoTime(),bgr.cols(),bgr.rows(),0).json());
                        assertEquals(settings.textDetectionMode.name(),json.getJSONObject("parameters").getString("text_detection_mode"));
                        assertEquals(ocr.usesMlKit() ? (run == 0 ? 1 : 4) : 0,
                                json.getJSONObject("ocr_engine").getInt("whole_image_sdk_calls"));
                        assertEquals(run == 0 ? 1 : 4,json.getJSONObject("timings").getJSONArray("text_detection_passes").length());
                        outputs.put(json); gate.finish(ticket);
                        try (FileOutputStream out = new FileOutputStream(new File(context.getFilesDir(),"four-direction-results.json"))) {
                            out.write(outputs.toString(2).getBytes(StandardCharsets.UTF_8));
                        }
                    }
                }
            }
        } finally { worker.shutdown(); bitmap.recycle(); rgba.release(); bgr.release(); }
    }

    @Test public void menuDefaultsToSingleAndPersistsBothChoicesAcrossRecreationAndLaunch() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        android.content.SharedPreferences prefs = context.getSharedPreferences("MainActivity",Context.MODE_PRIVATE);
        String previous = prefs.getString("textDetectionMode",null);
        prefs.edit().remove("textDetectionMode").commit();
        try {
            try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
                scenario.onActivity(activity -> {
                    assertEquals(TextDetectionMode.SINGLE,activity.textDetectionModeForTest());
                    activity.setTextDetectionModeForTest(TextDetectionMode.FOUR_DIRECTIONS);
                });
                InstrumentationRegistry.getInstrumentation().waitForIdleSync();
                scenario.recreate();
                scenario.onActivity(activity -> assertEquals(TextDetectionMode.FOUR_DIRECTIONS,activity.textDetectionModeForTest()));
            }
            try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
                scenario.onActivity(activity -> {
                    assertEquals(TextDetectionMode.FOUR_DIRECTIONS,activity.textDetectionModeForTest());
                    activity.setTextDetectionModeForTest(TextDetectionMode.SINGLE);
                });
                InstrumentationRegistry.getInstrumentation().waitForIdleSync();
            }
            try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
                scenario.onActivity(activity -> assertEquals(TextDetectionMode.SINGLE,activity.textDetectionModeForTest()));
            }
        } finally {
            if (previous == null) prefs.edit().remove("textDetectionMode").commit();
            else prefs.edit().putString("textDetectionMode",previous).commit();
        }
    }
}
