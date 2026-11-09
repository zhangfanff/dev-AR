package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import java.util.Arrays;
import java.util.Collections;
import static org.junit.Assert.*;

public class DownwardMatchingTest {
    private Detection point(int id, double x, double y, double confidence, String text) {
        return new Detection(id, confidence, new double[]{x-2,y-2,x+2,y+2}, null, text);
    }
    @Test public void closerSideTextsDoNotStealDownwardLabel() {
        Detection knob = point(0,100,100,.9,null);
        MatchReport r = GreedyMatcher.match(500,500,Collections.singletonList(knob),Arrays.asList(
                point(0,140,101,.9,"右侧"), point(1,60,101,.9,"左侧"), point(2,110,160,.8,"下方")),100,Double.NaN);
        assertEquals(2, r.matches.get(0).text.id);
        assertTrue(r.decisions.get(0).candidates.get(0).reasons.contains("outside_downward_sector"));
        assertTrue(r.decisions.get(0).candidates.get(1).reasons.contains("outside_downward_sector"));
    }
    @Test public void photographedRightHandVerticalTextIsRejected() {
        Detection knob = new Detection(0,.4059,new double[]{676.8803,155.9019,859.8505,346.255},null,null);
        Detection side = new Detection(2,0,new double[]{867,190,930,314},null,"侧面");
        Detection label = new Detection(6,0,new double[]{708,372,834,401},null,"标签");
        MatchReport r = GreedyMatcher.match(1706,1279,Collections.singletonList(knob),Arrays.asList(side,label),256,Double.NaN);
        assertEquals(6,r.matches.get(0).text.id);
        assertTrue(r.decisions.get(0).candidates.get(0).distance < r.matches.get(0).distance);
        assertFalse(r.decisions.get(0).candidates.get(0).geometric);
    }
    @Test public void labelInsideOversizedKnobBoxIsRejectedByRequiredOutsideCenterRule() {
        Detection knob = new Detection(1,.93,new double[]{496.7831,909.019,683.343,1145.4126},null,null);
        Detection label = new Detection(15,0,new double[]{523,1099,664,1123},null,"标签");
        assertTrue(label.cy < knob.box[3]);
        MatchReport r = GreedyMatcher.match(1706,1279,Collections.singletonList(knob),Collections.singletonList(label),256,Double.NaN);
        assertTrue(r.matches.isEmpty());
        assertTrue(r.decisions.get(0).candidates.get(0).reasons.contains("text_center_inside_knob_box"));
    }
    @Test public void boundaryAllowsEqualOffsetsButRejectsLargerSideOffset() {
        Detection knob = point(0,100,100,.9,null);
        assertEquals(1,GreedyMatcher.match(500,500,Collections.singletonList(knob),
                Collections.singletonList(point(0,130,130,.9,"边界")),100,Double.NaN).matches.size());
        assertEquals(0,GreedyMatcher.match(500,500,Collections.singletonList(knob),
                Collections.singletonList(point(0,130.01,130,.9,"超出")),100,Double.NaN).matches.size());
    }
    @Test public void narrowerSectorAllowsVerticalAndRejectsDiagonal() {
        Detection knob = point(0,100,100,.9,null);
        MatchReport r = GreedyMatcher.match(500,500,Collections.singletonList(knob),Arrays.asList(
                point(0,130,140,.9,"斜下方"),point(1,100,160,.9,"正下方")),100,Double.NaN,30);
        assertEquals(1,r.matches.get(0).text.id); assertEquals(30,r.maxAngleFromDownwardDegrees,0);
        assertEquals(0,GreedyMatcher.match(500,500,Collections.singletonList(knob),
                Collections.singletonList(point(0,101,140,.9,"偏一点")),100,Double.NaN,0).matches.size());
    }
    @Test public void invalidDirectionsDoNotConsumeTextForLowerConfidenceKnob() {
        MatchReport r = GreedyMatcher.match(500,500,Arrays.asList(point(0,100,100,.95,null),point(1,140,80,.4,null)),
                Collections.singletonList(point(0,140,101,.9,"标签")),100,Double.NaN);
        assertEquals(1,r.matches.size()); assertEquals(1,r.matches.get(0).knob.id);
        assertEquals("no_text_within_threshold",r.decisions.get(0).reason);
    }
    @Test public void sideOnlyProducesNoRecognitionAssignment() {
        MatchReport r = GreedyMatcher.match(500,500,Collections.singletonList(point(0,100,100,.9,null)),
                Collections.singletonList(point(0,140,101,.9,"侧面")),100,Double.NaN);
        assertTrue(RecognitionSelection.selected(r).isEmpty()); assertEquals(1,r.unmatchedTexts.size());
    }
    @Test(expected=IllegalArgumentException.class) public void wideSectorIsRejected() {
        GreedyMatcher.match(500,500,Collections.emptyList(),Collections.emptyList(),100,Double.NaN,61);
    }
    @Test(expected=IllegalArgumentException.class) public void nonFiniteSectorIsRejected() {
        GreedyMatcher.match(500,500,Collections.emptyList(),Collections.emptyList(),100,Double.NaN,Double.NaN);
    }
}
