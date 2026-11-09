package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import static org.junit.Assert.*;

public class MatchingTest {
    private Detection d(int id, double x, double y, double confidence, String text) {
        return new Detection(id, confidence, new double[]{x - 2, y - 2, x + 2, y + 2}, null, text);
    }
    @Test public void confidenceWinsOverNearerLowerConfidenceKnob() {
        MatchReport r = GreedyMatcher.match(500, 500, Arrays.asList(d(0, 100, 100, .4, null), d(1, 140, 100, .95, null)),
                Collections.singletonList(d(0, 100, 150, .9, "标签")), 100, Double.NaN);
        assertEquals(1, r.matches.size()); assertEquals(1, r.matches.get(0).knob.id);
        assertEquals("eligible_texts_already_used", r.decisions.get(1).reason);
    }
    @Test public void nearestTextWinsAndThenNextUnusedText() {
        MatchReport r = GreedyMatcher.match(500, 500, Arrays.asList(d(0, 100, 100, .9, null), d(1, 100, 105, .8, null)),
                Arrays.asList(d(1, 100, 160, .99, "远"), d(0, 100, 130, .5, "近")), 100, Double.NaN);
        assertEquals(2, r.matches.size()); assertEquals(0, r.matches.get(0).text.id); assertEquals(1, r.matches.get(1).text.id);
    }
    @Test public void boundaryDisplacementAndAboveRules() {
        MatchReport r = GreedyMatcher.match(500, 500, Collections.singletonList(d(0, 100, 100, .9, null)),
                Arrays.asList(d(0, 100, 90, .9, "上"), d(1, 130, 140, .9, "偏")), 50, Double.NaN);
        assertEquals(1, r.matches.size()); assertEquals(50, r.matches.get(0).distance, 1e-9); assertEquals(1, r.unmatchedTexts.size());
        assertEquals(0, GreedyMatcher.match(500, 500, r.knobs, r.texts, 49.99, Double.NaN).matches.size());
        assertEquals(0, GreedyMatcher.match(500, 500, r.knobs, r.texts, 100, 29).matches.size());
    }
    @Test public void emptyAndUnequalLists() {
        assertEquals(0, GreedyMatcher.match(500, 500, Collections.emptyList(), Collections.emptyList(), 100, Double.NaN).matches.size());
        MatchReport r = GreedyMatcher.match(500, 500, Collections.emptyList(), Collections.singletonList(d(0, 100, 100, .9, "孤立")), 100, Double.NaN);
        assertEquals(1, r.unmatchedTexts.size());
        r = GreedyMatcher.match(500, 500, Collections.singletonList(d(0, 100, 100, .9, null)), Collections.emptyList(), 100, Double.NaN);
        assertEquals("no_texts", r.decisions.get(0).reason);
    }
    @Test public void deterministicTies() {
        MatchReport r = GreedyMatcher.match(500, 500, Arrays.asList(d(3, 100, 100, .9, null), d(1, 100, 100, .9, null)),
                Arrays.asList(d(5, 90, 120, .8, "左"), d(2, 110, 120, .9, "右")), 100, Double.NaN);
        assertEquals(1, r.matches.get(0).knob.id); assertEquals(2, r.matches.get(0).text.id);
    }
    @Test(expected = IllegalArgumentException.class) public void invalidThresholdRejected() {
        GreedyMatcher.match(500, 500, Collections.emptyList(), Collections.emptyList(), Double.NaN, Double.NaN);
    }
    @Test public void ctcCollapsesRunsButPreservesBlankSeparatedRepeats() {
        FloatBuffer p = FloatBuffer.wrap(new float[]{.01f,.98f,.01f, .01f,.98f,.01f, .99f,.005f,.005f,
                .01f,.98f,.01f, .01f,.01f,.98f, .01f,.01f,.98f});
        TensorDecoders.Recognition r = TensorDecoders.ctc(p, 6, 3, Arrays.asList("", "合", "闸"));
        assertEquals("合合闸", r.text); assertEquals(.98, r.confidence, 1e-5);
    }
    @Test public void yoloNmsAndLetterboxCoordinates() {
        // Two overlapping detections; 1280 input had x=40,y=30 padding and scale=2.
        FloatBuffer p = FloatBuffer.wrap(new float[]{240,242, 230,232, 80,80, 100,100, .9f,.8f});
        java.util.List<Detection> knobs = TensorDecoders.yolo(p, 5, 2, 500, 500, 2, 40, 30, .25, .7);
        assertEquals(1, knobs.size()); assertArrayEquals(new double[]{80,75,120,125}, knobs.get(0).box, 1e-8);
    }
}
