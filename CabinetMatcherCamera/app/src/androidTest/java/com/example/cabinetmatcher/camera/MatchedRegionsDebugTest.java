package com.example.cabinetmatcher.camera;

import android.graphics.*;
import android.util.Size;
import android.view.View;
import androidx.annotation.OptIn;
import androidx.camera.view.TransformExperimental;
import androidx.camera.view.transform.OutputTransform;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.example.cabinetmatcher.camera.core.*;
import org.json.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.*;
import java.util.*;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
@OptIn(markerClass=TransformExperimental.class)
public class MatchedRegionsDebugTest {
    private static Detection region(int id,double x,double y) {
        double[][] polygon={{x,y},{x+100,y},{x+100,y+40},{x,y+40}};
        return new Detection(id,0,new double[]{x,y,x+100,y+40},polygon,null);
    }
    @Test public void failedOcrStillShowsActualRawRegionsArrowsAndUnassignedDetections()throws Exception {
        List<Detection> knobs=new ArrayList<>(),regions=new ArrayList<>();
        for(int i=0;i<3;i++) {
            double y=400+i*250;
            knobs.add(new Detection(i,.95-i*.01,new double[]{80,y,140,y+40},null,null));
            regions.add(region(i,240,y));
        }
        knobs.add(new Detection(3,.8,new double[]{80,1230,140,1270},null,null));
        regions.add(region(9,380,400)); // A farther alternative must not replace the low-confidence assignment.
        regions.add(region(10,650,1200));
        MatchReport report=GreedyMatcher.match(800,1400,knobs,regions,350,Double.NaN,30,SearchDirection.RIGHT);
        assertEquals(3,report.matches.size());
        RecognitionSelection.record(regions.get(0),"原始低分输出",.12,.5);
        RecognitionSelection.record(regions.get(1),"",0,.5);
        RecognitionSelection.failed(regions.get(2),new IllegalStateException("synthetic OCR failure"));
        assertEquals(0,report.recognizedMatches());assertEquals(3,report.matches.size());
        assertEquals(0,report.matches.get(0).text.id);assertEquals("not_requested",regions.get(3).recognitionStatus);
        Bitmap source=Bitmap.createBitmap(800,1400,Bitmap.Config.ARGB_8888);
        source.eraseColor(Color.WHITE);Canvas raw=new Canvas(source);Paint fill=new Paint();
        int[] colors={0xff1166aa,0xff228844,0xffaa2277};
        for(int i=0;i<3;i++) {fill.setColor(colors[i]);Detection region=regions.get(i);raw.drawRect((float)region.box[0],(float)region.box[1],(float)region.box[2],(float)region.box[3],fill);}
        CabinetPipeline.Settings settings=new CabinetPipeline.Settings();settings.searchDirection=SearchDirection.RIGHT;
        FrameGate gate=new FrameGate();gate.start();FrameGate.Ticket ticket=gate.tryBegin(System.nanoTime());
        MainActivity.FrameResult result=new MainActivity.FrameResult(ticket,report,null,settings,1,System.nanoTime(),800,1400,0);
        result.matchingSource=source;result.analysisTransform=new OutputTransform(new Matrix(),new Size(800,1400));
        JSONObject json=new JSONObject(result.json());
        assertTrue(json.getJSONObject("matching_debug").getBoolean("raw_region_preview_available"));
        assertEquals(3,json.getJSONArray("matches").length());
        assertEquals("low_confidence",json.getJSONArray("matches").getJSONObject(0).getString("recognition_status"));
        assertEquals("原始低分输出",json.getJSONArray("matches").getJSONObject(0).getJSONObject("text_region").getString("text"));
        assertEquals(.12,json.getJSONArray("matches").getJSONObject(0).getJSONObject("text_region").getDouble("confidence"),0);
        assertEquals("empty_text",json.getJSONArray("matches").getJSONObject(1).getString("recognition_status"));
        assertEquals("error",json.getJSONArray("matches").getJSONObject(2).getString("recognition_status"));
        assertEquals(1,json.getJSONArray("unmatched_knobs").length());
        InstrumentationRegistry.getInstrumentation().runOnMainSync(()->{
            android.content.Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
            MatchedRegionsView view=new MatchedRegionsView(context);view.show(report,source,.5);
            view.measure(View.MeasureSpec.makeMeasureSpec(800,View.MeasureSpec.EXACTLY),View.MeasureSpec.makeMeasureSpec(0,View.MeasureSpec.UNSPECIFIED));
            view.layout(0,0,800,view.getMeasuredHeight());
            Bitmap preview=Bitmap.createBitmap(800,view.getMeasuredHeight(),Bitmap.Config.ARGB_8888);
            try {
                view.draw(new Canvas(preview));int[] pixels=new int[preview.getWidth()*preview.getHeight()];
                preview.getPixels(pixels,0,preview.getWidth(),0,0,preview.getWidth(),preview.getHeight());
                for(int color:colors) {int count=0;for(int pixel:pixels)if(pixel==color)count++;assertTrue("Each failed OCR assignment must render its raw image",count>200);}
                save(preview,"match-debug-regions.png");
                LiveOverlayView overlay=new LiveOverlayView(context,()->new OutputTransform(new Matrix(),new Size(800,1400)));
                overlay.layout(0,0,800,1400);overlay.show(result,1);
                Bitmap live=Bitmap.createBitmap(800,1400,Bitmap.Config.ARGB_8888);
                try {
                    overlay.draw(new Canvas(live));
                    for(int i=0;i<3;i++)assertTrue("Failed OCR still draws an assignment arrow",Color.alpha(live.getPixel(200,420+i*250))>0);
                    assertTrue("Unselected text region must remain visible",Color.alpha(live.getPixel(650,1220))>0);
                    save(live,"match-debug-overlay.png");
                } finally {live.recycle();}
                Bitmap annotated=Annotations.draw(source,report);
                try {save(annotated,"match-debug-annotated.png");}finally{annotated.recycle();}
                view.show(null,null,0);
            } finally {preview.recycle();}
        });
        try(FileOutputStream file=new FileOutputStream(new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getFilesDir(),"match-debug-results.json"))) {
            file.write(json.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
        } finally {result.recycleImages();gate.finish(ticket);}
    }
    @Test public void bitmapCleanupReleasesRawAndAnnotatedFramesAndIsIdempotent() {
        FrameGate gate=new FrameGate();gate.start();FrameGate.Ticket ticket=gate.tryBegin(System.nanoTime());
        Bitmap raw=Bitmap.createBitmap(20,20,Bitmap.Config.ARGB_8888),annotated=raw.copy(Bitmap.Config.ARGB_8888,true);
        MainActivity.FrameResult frame=new MainActivity.FrameResult(ticket,new MatchReport(20,20,Collections.emptyList(),Collections.emptyList(),10,Double.NaN),annotated,new CabinetPipeline.Settings(),1,2,20,20,0);
        frame.matchingSource=raw;frame.recycleImages();frame.recycleImages();
        assertTrue(raw.isRecycled());assertTrue(annotated.isRecycled());gate.finish(ticket);
    }
    private static void save(Bitmap bitmap,String name) {
        try(FileOutputStream file=new FileOutputStream(new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getFilesDir(),name))) {
            assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG,100,file));
        }catch(IOException error){throw new RuntimeException(error);}
    }
}
