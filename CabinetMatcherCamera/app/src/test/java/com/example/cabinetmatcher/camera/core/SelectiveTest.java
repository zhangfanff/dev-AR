package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import java.util.Arrays;
import java.util.Collections;
import static org.junit.Assert.*;

public class SelectiveTest {
    private Detection knob(int id, double confidence) {
        return new Detection(id, confidence, new double[]{90,90,110,110}, null, null);
    }
    private Detection region(int id, double x, double y) {
        double[][] p = {{x-2,y-2},{x+2,y-2},{x+2,y+2},{x-2,y+2}};
        return new Detection(id, 0, new double[]{x-2,y-2,x+2,y+2}, p, null);
    }
    @Test public void onlyAssignedRegionsSelectedInKnobConfidenceOrder() {
        Detection close = region(1,100,130), far = region(2,100,170), above = region(0,100,50);
        MatchReport report = GreedyMatcher.match(500,500, Arrays.asList(knob(0,.5),knob(1,.9)),
                Arrays.asList(above,far,close), 100, Double.NaN);
        assertEquals(Arrays.asList(close,far), RecognitionSelection.selected(report));
        assertEquals(1, report.unmatchedTexts.size()); assertFalse(above.recognitionAttempted);
    }
    @Test public void failureDoesNotReassignOrRunUnusedRegion() {
        Detection close = region(1,100,120), far = region(2,100,140);
        MatchReport report = GreedyMatcher.match(500,500, Collections.singletonList(knob(0,.9)),
                Arrays.asList(close,far), 100, Double.NaN);
        RecognitionSelection.record(close,"模糊",.2,.5);
        assertEquals("low_confidence",close.recognitionStatus); assertEquals(0,report.recognizedMatches());
        assertSame(close,report.matches.get(0).text); assertFalse(far.recognitionAttempted);
        RecognitionSelection.record(close,"",0,.5); assertEquals("empty_text",close.recognitionStatus);
        RecognitionSelection.failed(close,new IllegalArgumentException("test"));
        assertEquals("error",close.recognitionStatus); assertTrue(close.recognitionAttempted);
        assertSame(close,report.matches.get(0).text); assertFalse(far.recognitionAttempted);
    }
    @Test public void noKnobsOrNoEligibleRegionMeansNoRecognition() {
        Detection above = region(0,100,80);
        MatchReport report = GreedyMatcher.match(500,500, Collections.emptyList(), Collections.singletonList(above),100,Double.NaN);
        assertTrue(RecognitionSelection.selected(report).isEmpty()); assertFalse(above.recognitionAttempted);
        report = GreedyMatcher.match(500,500,Collections.singletonList(knob(0,.9)),Collections.singletonList(above),100,Double.NaN);
        assertTrue(RecognitionSelection.selected(report).isEmpty());
    }
    @Test public void geometricTiesUseRegionIdNotFutureRecognitionConfidence() {
        Detection left = region(2,90,120), right = region(5,110,120);
        right.confidence=.99;
        MatchReport report = GreedyMatcher.match(500,500,Collections.singletonList(knob(0,.9)),Arrays.asList(right,left),100,Double.NaN);
        assertSame(left,report.matches.get(0).text);
    }
}
