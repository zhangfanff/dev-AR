package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import static org.junit.Assert.*;

public class TextDetectionSizeModeTest {
    private int[] size(TextDetectionSizeMode mode, int w, int h, int cap, boolean paddle) {
        return TextDetectionSizeMode.inputSize(w,h,mode.longSideLimit(w,h,cap),paddle);
    }
    @Test public void manualModeKeepsOriginalResizeAndAlignmentRules() {
        assertArrayEquals(new int[]{1600,896},size(TextDetectionSizeMode.MANUAL_LIMIT,1920,1080,1600,true));
        assertArrayEquals(new int[]{1600,900},size(TextDetectionSizeMode.MANUAL_LIMIT,1920,1080,1600,false));
        assertArrayEquals(new int[]{2048,1152},size(TextDetectionSizeMode.MANUAL_LIMIT,3840,2160,2048,true));
    }
    @Test public void followModeRetainsFullHdWithBackendSpecificAlignment() {
        assertArrayEquals(new int[]{1920,1088},size(TextDetectionSizeMode.FOLLOW_FRAME,1920,1080,1600,true));
        assertArrayEquals(new int[]{1920,1080},size(TextDetectionSizeMode.FOLLOW_FRAME,1920,1080,1600,false));
    }
    @Test public void followModeDoesNotApplyManual2048Ceiling() {
        assertArrayEquals(new int[]{3840,2176},size(TextDetectionSizeMode.FOLLOW_FRAME,3840,2160,2048,true));
        assertArrayEquals(new int[]{3840,2160},size(TextDetectionSizeMode.FOLLOW_FRAME,3840,2160,2048,false));
    }
    @Test public void portraitAndQuarterTurnsUseSwappedDimensions() {
        assertArrayEquals(new int[]{1088,1920},size(TextDetectionSizeMode.FOLLOW_FRAME,1080,1920,1600,true));
        assertArrayEquals(new int[]{1080,1920},size(TextDetectionSizeMode.FOLLOW_FRAME,1080,1920,1600,false));
        assertArrayEquals(new int[]{896,1600},size(TextDetectionSizeMode.MANUAL_LIMIT,1080,1920,1600,true));
    }
    @Test public void smallFramesAreNotEnlargedToConfiguredLongSide() {
        assertArrayEquals(new int[]{640,480},size(TextDetectionSizeMode.FOLLOW_FRAME,640,480,2048,true));
        assertArrayEquals(new int[]{640,480},size(TextDetectionSizeMode.MANUAL_LIMIT,640,480,2048,true));
    }
    @Test public void unknownOrMissingSavedModePreservesManualDefault() {
        assertEquals(TextDetectionSizeMode.MANUAL_LIMIT,TextDetectionSizeMode.fromStoredId(null));
        assertEquals(TextDetectionSizeMode.MANUAL_LIMIT,TextDetectionSizeMode.fromStoredId("future"));
        assertEquals(TextDetectionSizeMode.FOLLOW_FRAME,TextDetectionSizeMode.fromStoredId("FOLLOW_FRAME"));
    }
    @Test public void followModeIgnoresInactiveManualLimitWhileManualModeValidatesIt() {
        CabinetPipeline.Settings settings = new CabinetPipeline.Settings();
        settings.detSide = 3000;
        try { settings.validate(); fail("Manual limit must remain bounded"); }
        catch (IllegalArgumentException expected) { assertTrue(expected.getMessage().contains("2048")); }
        settings.textDetectionSizeMode = TextDetectionSizeMode.FOLLOW_FRAME;
        settings.validate();
    }
}
