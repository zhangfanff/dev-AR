package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import java.util.Arrays;
import java.util.Collections;
import static org.junit.Assert.*;

public class OutsideKnobCenterTest {
    private Detection knob() { return new Detection(0,.9,new double[]{80,80,120,120},null,null); }
    private Detection text(int id,double x,double y) {
        return new Detection(id,0,new double[]{x-1,y-1,x+1,y+1},null,"标签");
    }
    private MatchReport match(SearchDirection direction,Detection... texts) {
        return GreedyMatcher.match(500,500,Collections.singletonList(knob()),Arrays.asList(texts),100,Double.NaN,45,direction);
    }
    @Test public void closerInsideTextCannotStealOutsideLabel() {
        MatchReport report=match(SearchDirection.DOWN,text(0,100,110),text(1,100,140));
        assertEquals(1,report.matches.size()); assertEquals(1,report.matches.get(0).text.id);
        MatchReport.Candidate rejected=report.decisions.get(0).candidates.get(0);
        assertFalse(rejected.geometric); assertFalse(rejected.selected);
        assertTrue(rejected.reasons.contains("text_center_inside_knob_box"));
    }
    @Test public void allEdgesAndCornersAreInsideAndRejected() {
        double[][] boundary={{80,100},{120,100},{100,80},{100,120},{80,80},{120,80},{80,120},{120,120}};
        for(double[] p:boundary) {
            MatchReport report=match(SearchDirection.ALL,text(0,p[0],p[1]));
            assertTrue(report.matches.isEmpty());
            assertEquals("no_text_outside_knob_box",report.decisions.get(0).reason);
        }
    }
    @Test public void fractionalCenterJustOutsideIsEligible() {
        double[][] outside={{79.999,100},{120.001,100},{100,79.999},{100,120.001}};
        for(double[] p:outside) assertEquals(1,match(SearchDirection.ALL,text(0,p[0],p[1])).matches.size());
    }
    @Test public void overlappingTextBoxIsAllowedWhenItsCenterIsOutside() {
        Detection overlapping=new Detection(0,0,new double[]{90,115,110,135},null,"重叠但中心在外");
        assertTrue(overlapping.box[1]<knob().box[3]);
        assertEquals(1,match(SearchDirection.DOWN,overlapping).matches.size());
    }
    @Test public void rejectionIsRelativeToCurrentKnobAndDoesNotConsumeText() {
        Detection high=knob(),low=new Detection(1,.5,new double[]{90,40,110,60},null,null);
        MatchReport report=GreedyMatcher.match(500,500,Arrays.asList(high,low),Collections.singletonList(text(0,100,110)),100,Double.NaN,45,SearchDirection.DOWN);
        assertEquals(1,report.matches.size()); assertEquals(low.id,report.matches.get(0).knob.id);
        assertEquals("no_text_outside_knob_box",report.decisions.get(0).reason);
    }
    @Test public void outsideRuleAppliesToEveryDirectionIncludingFourWay() {
        double[][] inside={{100,110},{100,90},{90,100},{110,100},{100,110}};
        double[][] outside={{100,140},{100,60},{60,100},{140,100},{100,140}};
        for(SearchDirection direction:SearchDirection.values()) {
            int i=direction.ordinal(); MatchReport report=match(direction,text(0,inside[i][0],inside[i][1]),text(1,outside[i][0],outside[i][1]));
            assertEquals(1,report.matches.size()); assertEquals(1,report.matches.get(0).text.id);
        }
    }
    @Test public void insideOnlyTextsAreUnmatchedAndNeverSelectedForOcr() {
        double[][] polygon={{95,105},{105,105},{105,115},{95,115}};
        Detection region=new Detection(0,0,new double[]{95,105,105,115},polygon,null);
        MatchReport report=match(SearchDirection.DOWN,region);
        assertEquals(1,report.unmatchedTexts.size()); assertTrue(RecognitionSelection.selected(report).isEmpty());
        assertFalse(report.texts.get(0).recognitionAttempted);
    }
    @Test public void usesPolygonCenterRatherThanRejectingByBoxIntersectionOrBoxMidpoint() {
        double[][] polygon={{110,80},{111,80},{140,120},{110,120}};
        Detection region=new Detection(0,0,new double[]{110,80,140,120},polygon,null);
        assertTrue((region.box[0]+region.box[2])/2>120); assertTrue(region.cx<120);
        assertTrue(match(SearchDirection.ALL,region).matches.isEmpty());
    }
}
