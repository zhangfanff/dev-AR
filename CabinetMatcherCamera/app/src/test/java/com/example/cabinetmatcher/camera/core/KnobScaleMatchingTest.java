package com.example.cabinetmatcher.camera.core;

import java.util.*;
import org.junit.Test;
import static org.junit.Assert.*;

public class KnobScaleMatchingTest {
    private Detection knob(int id,double x1,double y1,double x2,double y2,double score) {
        return new Detection(id,score,new double[]{x1,y1,x2,y2},null,null);
    }
    private Detection text(int id,double x1,double y1,double x2,double y2) {
        return new Detection(id,0,new double[]{x1,y1,x2,y2},new double[][]{{x1,y1},{x2,y1},{x2,y2},{x1,y2}},null);
    }
    private MatchReport match(Detection k,Detection t) {
        return GreedyMatcher.matchByKnobSize(1000,1000,Collections.singletonList(k),Collections.singletonList(t),45,SearchDirection.RIGHT);
    }
    @Test public void exactKnobDiagonalAcceptedAndJustBeyondRejected() {
        Detection k=knob(0,100,100,160,180,.9);
        MatchReport boundary=match(k,text(0,225,135,235,145));
        assertEquals(1,boundary.matches.size());assertEquals(100,boundary.maxDistanceFor(k),1e-9);
        MatchReport beyond=match(k,text(0,225.01,135,235.01,145));
        assertTrue(beyond.matches.isEmpty());assertTrue(beyond.decisions.get(0).candidates.get(0).reasons.contains("distance_exceeded"));
    }
    @Test public void zoomScalesDistanceAndLimitTogether() {
        for(int scale:new int[]{1,2,3}) {
            Detection k=knob(0,100*scale,100*scale,160*scale,180*scale,.9);
            MatchReport report=match(k,text(0,205*scale,135*scale,215*scale,145*scale));
            assertEquals(1,report.matches.size());assertEquals(.8,report.matches.get(0).distance/report.maxDistanceFor(k),1e-9);
        }
    }
    @Test public void imageCanvasSizeDoesNotChangeAssignments() {
        Detection k=knob(0,100,100,160,180,.9),t=text(0,225.01,135,235.01,145);
        for(int size:new int[]{300,500,5000}) {
            MatchReport report=GreedyMatcher.matchByKnobSize(size,size,Collections.singletonList(k),Collections.singletonList(t),45,SearchDirection.RIGHT);
            assertTrue(report.matches.isEmpty());assertEquals(100,report.maxDistanceFor(k),1e-9);
        }
    }
    @Test public void eachKnobHasItsOwnDistanceLimit() {
        Detection small=knob(0,100,100,120,120,.99),large=knob(1,70,70,150,150,.8),t=text(0,140,105,160,115);
        MatchReport report=GreedyMatcher.matchByKnobSize(300,300,Arrays.asList(small,large),Collections.singletonList(t),45,SearchDirection.RIGHT);
        assertEquals(1,report.matches.size());assertEquals(1,report.matches.get(0).knob.id);
        assertEquals(Math.hypot(20,20),report.decisions.get(0).candidates.get(0).maxDistance,1e-9);
        assertEquals(Math.hypot(80,80),report.decisions.get(1).candidates.get(0).maxDistance,1e-9);
    }
    @Test public void exactlyOneThirdOutsideAllowedEvenWithCenterInside() {
        Detection k=knob(0,100,100,200,200,.9),t=text(0,180,140,210,160);
        assertTrue(t.cx<k.box[2]);assertEquals(1.0/3,TextOverlap.outsideRatio(k,t),1e-9);
        assertEquals(1,match(k,t).matches.size());
    }
    @Test public void justBelowOneThirdOutsideRejected() {
        Detection k=knob(0,100,100,200,200,.9),t=text(0,179.99,140,209.99,160);
        MatchReport report=match(k,t);assertTrue(report.matches.isEmpty());
        assertTrue(report.decisions.get(0).candidates.get(0).reasons.contains("text_outside_area_insufficient"));
        assertEquals("no_text_with_required_outside_area",report.decisions.get(0).reason);
    }
    @Test public void rotatedTextUsesPolygonAreaRatherThanBoundingRectangle() {
        Detection k=knob(0,10,10,20,20,.9);
        Detection diamond=new Detection(0,0,new double[]{5,5,25,25},new double[][]{{15,5},{25,15},{15,25},{5,15}},null);
        assertEquals(.5,TextOverlap.outsideRatio(k,diamond),1e-9);
        assertEquals(.75,TextOverlap.outsideRatio(k,text(0,5,5,25,25)),1e-9);
    }
    @Test public void fullyContainedTextRejected() {
        Detection k=knob(0,100,100,200,200,.9),t=text(0,160,140,180,160);
        assertEquals(0,TextOverlap.outsideRatio(k,t),0);assertTrue(match(k,t).matches.isEmpty());
    }
    @Test public void zeroAreaTextRejected() {
        MatchReport report=match(knob(0,100,100,200,200,.9),text(0,180,150,210,150));
        assertTrue(report.matches.isEmpty());assertEquals(0,report.decisions.get(0).candidates.get(0).outsideRatio,0);
    }
    @Test public void touchingBoundaryHasNoPositiveOverlapArea() {
        Detection k=knob(0,100,100,200,200,.9),t=text(0,200,140,220,160);
        assertEquals(1,TextOverlap.outsideRatio(k,t),0);assertEquals(1,match(k,t).matches.size());
    }
    @Test public void priorityOneToOneAndFailedOcrDoNotReassign() {
        Detection high=knob(0,100,100,200,200,.95),low=knob(1,105,100,205,200,.8),t=text(0,210,140,230,160);
        MatchReport report=GreedyMatcher.matchByKnobSize(500,500,Arrays.asList(low,high),Collections.singletonList(t),45,SearchDirection.RIGHT);
        assertEquals(1,report.matches.size());assertEquals(0,report.matches.get(0).knob.id);
        assertEquals("eligible_texts_already_used",report.decisions.get(1).reason);
        RecognitionSelection.record(t,"低分",.1,.5);
        assertEquals(1,report.matches.size());assertEquals("low_confidence",report.matches.get(0).text.recognitionStatus);
    }
    @Test public void fourDirectionsKeepTheirSectorRestriction() {
        Detection k=knob(0,100,100,200,200,.9),right=text(0,210,140,230,160);
        for(SearchDirection direction:SearchDirection.values()) {
            MatchReport report=GreedyMatcher.matchByKnobSize(500,500,Collections.singletonList(k),Collections.singletonList(right),30,direction);
            assertEquals(direction==SearchDirection.RIGHT||direction==SearchDirection.ALL?1:0,report.matches.size());
        }
    }
}
