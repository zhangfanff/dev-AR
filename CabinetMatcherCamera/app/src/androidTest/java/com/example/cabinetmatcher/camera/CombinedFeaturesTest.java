package com.example.cabinetmatcher.camera;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.example.cabinetmatcher.camera.core.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collections;
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
public class CombinedFeaturesTest {
    @Test public void cabinetOrientationAndHandModelsRunInSameProcessOnSameImage() throws Exception {
        for(ModelChoice.Ocr ocr:ModelChoice.Ocr.values()) {
        android.content.Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertTrue(OpenCVLoader.initLocal()); Bitmap input;
        try(InputStream stream=context.getAssets().open("demo_cabinet.jpg")) { input=BitmapFactory.decodeStream(stream); }
        Mat rgba=new Mat(),bgr=new Mat(); Bitmap annotation=null;
        ModelChoice choice=new ModelChoice(640,ocr);
        try(CabinetPipeline pipeline=CabinetPipelines.create(ModelAssets.prepare(context),choice); HandEngine hand=new HandEngine(context)) {
            Utils.bitmapToMat(input,rgba); Imgproc.cvtColor(rgba,bgr,Imgproc.COLOR_RGBA2BGR);
            CabinetPipeline.Settings settings=new CabinetPipeline.Settings(); settings.models=choice;
            assertTrue(settings.angleClassifier); MatchReport report=pipeline.run(bgr,settings,message -> {});
            assertTrue(report.matches.size()>0); assertTrue(report.timing.angleClassifierEnabled);
            assertTrue(report.timing.nanos("text_loop.angle.inference")>0);
            for(Detection region:report.texts) assertNull(region.recognitionError);
            // Cabinet annotation is independent from MediaPipe input bitmap ownership.
            annotation=Annotations.draw(input,report);
            float[][] hands=hand.detect(input,1000);
            HandDrawing.draw(new Canvas(annotation),hands,report.width,report.height,new Matrix(),2);
            assertTrue(hands.length<=2);
            assertEquals(report.matches.size(),report.timing.recognitionModelCalls);
        } finally {
            if(!input.isRecycled()) input.recycle(); if(annotation!=null) annotation.recycle(); rgba.release(); bgr.release();
        }
        }
    }
    @Test public void realHandLandmarksExportWithCabinetFrameIdAndPixelCoordinates() throws Exception {
        android.content.Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        Bitmap input;
        try(InputStream stream=InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("right_hands.jpg")) {
            input=BitmapFactory.decodeStream(stream);
        }
        int width=input.getWidth(),height=input.getHeight(); Bitmap annotation=input.copy(Bitmap.Config.ARGB_8888,true);
        try(HandEngine hand=new HandEngine(context)) {
            float[][] hands=hand.detect(input,1000); assertTrue(hands.length>0);
            HandDrawing.draw(new Canvas(annotation),hands,width,height,new Matrix(),2);
            FrameGate gate=new FrameGate(); gate.start(); FrameGate.Ticket ticket=gate.tryBegin(System.nanoTime());
            MatchReport report=new MatchReport(width,height,Collections.emptyList(),Collections.emptyList(),100,Double.NaN);
            MainActivity.FrameResult frame=new MainActivity.FrameResult(ticket,report,annotation,new CabinetPipeline.Settings(),1234,System.nanoTime(),width,height,0);
            frame.hands=hands; frame.handDetectionCompleted=true; frame.handTimestampMs=1000;
            JSONObject output=new JSONObject(frame.json()); JSONObject actual=output.getJSONObject("hand_landmarks");
            assertTrue(actual.getBoolean("completed")); assertTrue(actual.getBoolean("same_frame_as_cabinet"));
            assertEquals(output.getJSONObject("video").getLong("frame_id"),actual.getLong("frame_id"));
            assertEquals(1234,actual.getLong("camera_timestamp_ns"));
            JSONArray points=actual.getJSONArray("hands").getJSONObject(0).getJSONArray("landmarks");
            assertEquals(21,points.length());
            JSONObject tip=points.getJSONObject(8);
            assertEquals(tip.getDouble("x_normalized")*width,tip.getDouble("x_px"),.001);
            assertEquals(tip.getDouble("y_normalized")*height,tip.getDouble("y_px"),.001);
            assertEquals("hand_landmarks_same_frame",output.getJSONArray("execution_order").getString(4));
            try(FileOutputStream file=new FileOutputStream(new File(context.getFilesDir(),"combined-hand-example.png"))) {
                assertTrue(annotation.compress(Bitmap.CompressFormat.PNG,100,file));
            }
            try(FileOutputStream file=new FileOutputStream(new File(context.getFilesDir(),"combined-hand-example.json"))) {
                file.write(output.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }
            gate.finish(ticket);
        } finally { if(!input.isRecycled()) input.recycle(); annotation.recycle(); }
    }
    @Test public void nonDownwardReportNamesCorrectAxisAndIncludesCandidateRejections() throws Exception {
        Detection knob=new Detection(0,.9,new double[]{90,90,110,110},null,null);
        Detection right=new Detection(0,.9,new double[]{130,95,150,105},null,"右方标签");
        Detection below=new Detection(1,.9,new double[]{95,130,105,150},null,"下方标签");
        MatchReport report=GreedyMatcher.match(300,300,Collections.singletonList(knob),Arrays.asList(right,below),100,Double.NaN,30,SearchDirection.RIGHT);
        JSONObject output=new JSONObject(ReportWriter.json(report));
        assertEquals("RIGHT",output.getJSONObject("parameters").getString("search_direction"));
        assertEquals(30,output.getJSONObject("parameters").getDouble("sector_half_angle_degrees"),0);
        assertTrue(output.getJSONObject("parameters").isNull("max_angle_from_downward_degrees"));
        assertEquals("RIGHT",output.getJSONArray("matches").getJSONObject(0).getString("search_axis"));
        assertTrue(output.getJSONArray("knob_results").getJSONObject(0).getJSONArray("candidate_evaluations")
                .getJSONObject(1).getJSONArray("rejection_reasons").length()>0);
    }
}
