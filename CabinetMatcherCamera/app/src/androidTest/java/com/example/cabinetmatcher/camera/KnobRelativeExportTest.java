package com.example.cabinetmatcher.camera;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.example.cabinetmatcher.camera.core.*;
import org.json.*;
import java.util.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class KnobRelativeExportTest {
    private MatchReport report(double x) {
        Detection k=new Detection(0,.9,new double[]{100,100,200,200},null,null);
        Detection t=new Detection(0,0,new double[]{x,140,x+30,160},new double[][]{{x,140},{x+30,140},{x+30,160},{x,160}},null);
        return GreedyMatcher.matchByKnobSize(500,500,Collections.singletonList(k),Collections.singletonList(t),45,SearchDirection.RIGHT);
    }
    @Test public void jsonExplainsPerKnobDistanceAndAreaInsteadOfCenterRule()throws Exception {
        JSONObject json=new JSONObject(ReportWriter.json(report(180))),p=json.getJSONObject("parameters");
        assertEquals("per_knob_box_diagonal",p.getString("distance_basis"));
        assertTrue(p.isNull("max_distance_ratio_of_diagonal"));assertTrue(p.isNull("max_distance_px"));
        assertEquals(1,p.getInt("max_distance_knob_diagonal_multiplier"));
        assertFalse(p.getBoolean("text_center_must_be_outside_knob_box"));
        assertEquals(1.0/3,p.getDouble("min_text_outside_knob_area_ratio"),1e-9);
        JSONObject pair=json.getJSONArray("matches").getJSONObject(0);
        assertEquals(Math.hypot(100,100),pair.getDouble("max_distance_processing_px"),1e-9);
        assertEquals(1.0/3,pair.getDouble("text_outside_knob_area_ratio"),1e-9);
    }
    @Test public void rejectedCandidatesExportMeasuredAreaAndDistanceLimit()throws Exception {
        JSONObject json=new JSONObject(ReportWriter.json(report(179.99)));
        assertEquals(0,json.getJSONArray("matches").length());
        JSONObject candidate=json.getJSONArray("knob_results").getJSONObject(0).getJSONArray("candidate_evaluations").getJSONObject(0);
        assertTrue(candidate.getDouble("text_outside_knob_area_ratio")<1.0/3);
        assertEquals("text_outside_area_insufficient",candidate.getJSONArray("rejection_reasons").getString(0));
        assertEquals(Math.hypot(100,100),candidate.getDouble("max_distance_processing_px"),1e-9);
    }
    @Test public void exportedLimitUsesScaledKnobDimensions()throws Exception {
        MatchReport report=report(180);report.sourceWidth=1000;report.sourceHeight=1500;
        JSONObject pair=new JSONObject(ReportWriter.json(report)).getJSONArray("matches").getJSONObject(0);
        assertEquals(Math.hypot(200,300),pair.getDouble("max_distance_px"),1e-9);
        assertEquals(1.0/3,pair.getDouble("text_outside_knob_area_ratio"),1e-9);
    }
}
