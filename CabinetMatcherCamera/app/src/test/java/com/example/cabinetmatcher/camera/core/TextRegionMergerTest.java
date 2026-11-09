package com.example.cabinetmatcher.camera.core;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import static org.junit.Assert.*;

public class TextRegionMergerTest {
    private static double[][] box(double x, double y, double width, double height) {
        return new double[][]{{x,y},{x+width,y},{x+width,y+height},{x,y+height}};
    }
    private static TextRegionMerger.Region region(double[][] polygon, int degrees) {
        return new TextRegionMerger.Region(polygon, degrees);
    }
    private static void assertPolygon(double[][] expected, double[][] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) assertArrayEquals(expected[i], actual[i], 1e-9);
    }
    @Test public void restoresEveryRotationOnNonSquareImageWithoutChangingCornerOrder() {
        double[][] original = box(10, 20, 30, 15);
        assertPolygon(original, TextRegionMerger.restore(original, 0, 200, 100));
        assertPolygon(original, TextRegionMerger.restore(new double[][]{{79,10},{79,40},{64,40},{64,10}}, 90, 200, 100));
        assertPolygon(original, TextRegionMerger.restore(new double[][]{{189,79},{159,79},{159,64},{189,64}}, 180, 200, 100));
        assertPolygon(original, TextRegionMerger.restore(new double[][]{{20,189},{20,159},{35,159},{35,189}}, 270, 200, 100));
    }
    @Test public void restoresImageBordersAndClipsToValidPixels() {
        assertPolygon(new double[][]{{0,99},{199,0}},
                TextRegionMerger.restore(new double[][]{{0,0},{99,199}},90,200,100));
        assertPolygon(new double[][]{{0,99},{199,0}},
                TextRegionMerger.restore(new double[][]{{-2,-3},{101,203}},90,200,100));
    }
    @Test public void polygonIouHandlesBothWindingsAndPartialOverlap() {
        double[][] a = box(0,0,20,10), b = box(10,0,20,10);
        assertEquals(1.0/3, TextRegionMerger.iou(a,b),1e-9);
        Collections.reverse(Arrays.asList(b));
        assertEquals(1.0/3,TextRegionMerger.iou(a,b),1e-9);
        assertEquals(1,TextRegionMerger.iou(a,a),1e-9);
        assertEquals(0,TextRegionMerger.iou(a,box(20,0,10,10)),0);
        assertEquals(0,TextRegionMerger.iou(a,box(0,0,0,10)),0);
    }
    @Test public void keepsNearbySlantedLinesWhoseBoundingRectanglesOverlap() {
        double[][] a = {{0,0},{100,100},{98,102},{-2,2}};
        double[][] b = {{0,10},{100,110},{98,112},{-2,12}};
        assertEquals(0,TextRegionMerger.iou(a,b),1e-9);
        assertEquals(2,TextRegionMerger.merge(Arrays.asList(region(a,0),region(b,90))).size());
    }
    @Test public void mergesCrossDirectionDuplicatesAndKeepsNewRegions() {
        double[][] original = box(10,20,100,20), newRegion = box(140,100,30,20);
        List<double[][]> result = TextRegionMerger.merge(Arrays.asList(region(original,0),
                region(box(11,21,98,18),90),region(original,180),region(newRegion,270)));
        assertEquals(2,result.size()); assertSame(original,result.get(0)); assertSame(newRegion,result.get(1));
    }
    @Test public void largerBoxWinsAndEqualAreaPrefersOriginalPass() {
        double[][] small = box(10,10,30,10), large = box(9,9,32,12);
        assertSame(large,TextRegionMerger.merge(Arrays.asList(region(small,0),region(large,90))).get(0));
        assertSame(small,TextRegionMerger.merge(Arrays.asList(region(small,0),region(small.clone(),90))).get(0));
    }
    @Test public void preservesSamePassSegmentationAndRejectsZeroArea() {
        assertEquals(2,TextRegionMerger.merge(Arrays.asList(region(box(10,10,30,10),0),
                region(box(11,10,30,10),0),region(box(10,10,30,10),90),region(box(0,0,0,10),270))).size());
        assertTrue(TextRegionMerger.merge(Collections.emptyList()).isEmpty());
    }
    @Test public void restoredRegionsStillMatchOriginalSearchDirectionOnce() {
        double[][] restored = TextRegionMerger.restore(box(39,100,10,30),90,300,150);
        List<double[][]> merged = TextRegionMerger.merge(Arrays.asList(region(restored,90),region(restored.clone(),270)));
        double[][] p = merged.get(0);
        Detection text = new Detection(0,0,new double[]{100,100,130,110},p,null);
        Detection knob = new Detection(0,.9,new double[]{90,50,130,90},null,null);
        MatchReport report = GreedyMatcher.matchByKnobSize(300,150,Arrays.asList(knob,
                new Detection(1,.8,knob.box,null,null)),Collections.singletonList(text),45,SearchDirection.DOWN);
        assertEquals(1,report.matches.size()); assertEquals(0,report.matches.get(0).knob.id);
    }
    @Test public void missingOrUnknownSavedModeUsesOriginalSingleDirection() {
        assertEquals(TextDetectionMode.SINGLE,TextDetectionMode.fromStoredId(null));
        assertEquals(TextDetectionMode.SINGLE,TextDetectionMode.fromStoredId("future_mode"));
        assertEquals(TextDetectionMode.FOUR_DIRECTIONS,TextDetectionMode.fromStoredId("FOUR_DIRECTIONS"));
    }
}
