package com.example.cabinetmatcher.camera;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.util.Size;
import androidx.annotation.OptIn;
import androidx.camera.view.TransformExperimental;
import androidx.camera.view.transform.CoordinateTransform;
import androidx.camera.view.transform.OutputTransform;
import androidx.camera.view.transform.ImageProxyTransformFactory;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.ImageInfo;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.example.cabinetmatcher.camera.core.*;
import java.util.Arrays;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
@OptIn(markerClass = TransformExperimental.class)
public class OverlayPreviewTest {
    @Test public void croppedFrameTransformHonorsAllFourRotations() {
        int[] rotations={0,90,180,270};
        float[][] expected={{0,0,640,0,640,480,0,480},{480,0,480,640,0,640,0,0},
                {640,480,0,480,0,0,640,0},{0,640,0,0,480,0,480,640}};
        ImageProxyTransformFactory factory=new ImageProxyTransformFactory();
        factory.setUsingCropRect(true); factory.setUsingRotationDegrees(true);
        for (int i=0;i<rotations.length;i++) {
            int rotation=rotations[i];
            ImageInfo info=(ImageInfo)java.lang.reflect.Proxy.newProxyInstance(ImageInfo.class.getClassLoader(),
                    new Class<?>[]{ImageInfo.class},(proxy,method,args) -> {
                        if (method.getName().equals("getRotationDegrees")) return rotation;
                        throw new UnsupportedOperationException(method.getName());
                    });
            ImageProxy frame=(ImageProxy)java.lang.reflect.Proxy.newProxyInstance(ImageProxy.class.getClassLoader(),
                    new Class<?>[]{ImageProxy.class},(proxy,method,args) -> {
                        switch(method.getName()) {
                            case "getImageInfo": return info;
                            case "getWidth": return 1920;
                            case "getHeight": return 1080;
                            case "getCropRect": return new android.graphics.Rect(50,120,690,600);
                            default: throw new UnsupportedOperationException(method.getName());
                        }
                    });
            float[] corners={-1,-1,1,-1,1,1,-1,1};
            factory.getOutputTransform(frame).getMatrix().mapPoints(corners);
            assertArrayEquals("rotation="+rotation,expected[i],corners,.001f);
        }
    }
    @Test public void transformedCoordinatesAccountForPreviewLetterboxing() {
        // Source is an upright, already-cropped 900x1600 analysis frame.
        // Normalized (-1..1) -> source, followed by FIT_CENTER into a 500x400 preview.
        Matrix source = new Matrix(); source.setScale(450,800); source.postTranslate(450,800);
        Matrix target = new Matrix(); target.setScale(112.5f,200); target.postTranslate(250,200);
        CoordinateTransform transform = new CoordinateTransform(
                new OutputTransform(source,new Size(1600,900)),new OutputTransform(target,new Size(1600,900)));
        float[] points={0,0,900,1600,450,800}; transform.mapPoints(points);
        assertArrayEquals(new float[]{137.5f,0,362.5f,400,250,200},points,.001f);
    }
    @Test public void overlayDrawsVectorResultsAndClearsWithoutUsingSnapshotBitmap() {
        InstrumentationRegistry.getInstrumentation().runOnMainSync(() -> {
            android.content.Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
            Matrix identity=new Matrix();
            LiveOverlayView overlay=new LiveOverlayView(context,() -> new OutputTransform(identity,new Size(400,600)));
            overlay.layout(0,0,400,600);
            Detection knob=new Detection(0,.9,new double[]{100,200,150,250},null,null);
            Detection region=new Detection(0,.9,new double[]{100,300,190,330},null,"1KK 开关");
            region.recognitionStatus="recognized";
            MatchReport report=new MatchReport(400,600,Arrays.asList(knob),Arrays.asList(region),100,Double.NaN);
            report.matches.add(new MatchReport.Pair(knob,region,0,90));
            FrameGate gate=new FrameGate(); gate.start(); FrameGate.Ticket ticket=gate.tryBegin(System.nanoTime());
            MainActivity.FrameResult result=new MainActivity.FrameResult(ticket,report,null,new CabinetPipeline.Settings(),0,System.nanoTime(),400,600,0);
            result.analysisTransform=new OutputTransform(identity,new Size(400,600));
            overlay.show(result,1); assertTrue(overlay.hasResultForTest());
            Bitmap bitmap=Bitmap.createBitmap(400,600,Bitmap.Config.ARGB_8888);
            overlay.draw(new Canvas(bitmap));
            // A knob corner lies outside the HUD and must be a colored vector stroke.
            assertTrue(Color.alpha(bitmap.getPixel(100,200))>0);
            assertTrue(Color.green(bitmap.getPixel(100,200))>Color.red(bitmap.getPixel(100,200)));
            overlay.clear(); assertFalse(overlay.hasResultForTest());
            bitmap.eraseColor(Color.TRANSPARENT); overlay.draw(new Canvas(bitmap));
            assertEquals(Color.TRANSPARENT,bitmap.getPixel(100,200));
            bitmap.recycle(); gate.finish(ticket);
        });
    }
    @Test public void exportExplainsDelayedLiveOverlayAndKeepsAnalysisCoordinates() throws Exception {
        MatchReport report=new MatchReport(400,600,java.util.Collections.emptyList(),java.util.Collections.emptyList(),100,Double.NaN);
        FrameGate gate=new FrameGate(); gate.start(); FrameGate.Ticket ticket=gate.tryBegin(System.nanoTime());
        MainActivity.FrameResult result=new MainActivity.FrameResult(ticket,report,null,new CabinetPipeline.Settings(),123,System.nanoTime(),600,400,90);
        JSONObject json=new JSONObject(result.json());
        JSONObject video=json.getJSONObject("video");
        assertEquals("live_preview_and_processed_frame",video.getString("result_overlay_target"));
        assertFalse(video.getJSONObject("live_overlay").getBoolean("motion_compensation"));
        assertFalse(video.getJSONObject("live_overlay").getBoolean("tracking"));
        assertEquals(400,json.getJSONObject("image_size").getInt("width"));
        assertEquals(90,video.getInt("rotation_degrees")); gate.finish(ticket);
    }
}
