package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import static org.junit.Assert.*;
import java.nio.FloatBuffer;
import java.util.List;

public class ModelChoiceTest {
    @Test public void choicesSelectPairedOcrFilesAndCorrectDictionary() {
        for (int side : new int[]{640, 1280}) for (ModelChoice.Ocr ocr : ModelChoice.Ocr.values()) {
            ModelChoice choice = new ModelChoice(side, ocr);
            assertEquals("knob_" + side + ".onnx", choice.yoloFile());
            assertEquals(choice, new ModelChoice(side, ocr));
            assertEquals(choice.hashCode(), new ModelChoice(side, ocr).hashCode());
            assertTrue(choice.label().contains(ocr.label));
            assertNotEquals(choice, new ModelChoice(side == 640 ? 1280 : 640, ocr));
        }
        assertEquals(6625, ModelChoice.Ocr.V4.classes);
        assertEquals(6906, ModelChoice.Ocr.V6_TINY.classes);
        assertNotEquals(ModelChoice.Ocr.V4.dictionaryFile, ModelChoice.Ocr.V6_TINY.dictionaryFile);
        assertEquals(new ModelChoice(1280, ModelChoice.Ocr.V4), ModelChoice.defaults());
    }
    @Test(expected = IllegalArgumentException.class) public void invalidInputSizeRejected() { new ModelChoice(960, ModelChoice.Ocr.V4); }
    @Test(expected = IllegalArgumentException.class) public void nullOcrRejected() { new ModelChoice(640, null); }
    @Test public void crnnHasItsOwnInputAndMlKitUsesSdkRatherThanOnnxAssets() {
        assertEquals(32, ModelChoice.Ocr.DBNET_CRNN.recognitionHeight());
        assertEquals(6625, ModelChoice.Ocr.DBNET_CRNN.classes);
        assertNotEquals(ModelChoice.Ocr.V4.recognitionFile, ModelChoice.Ocr.DBNET_CRNN.recognitionFile);
        assertEquals(ModelChoice.Ocr.V4.detectionFile, ModelChoice.Ocr.DBNET_CRNN.detectionFile);
        assertTrue(ModelChoice.Ocr.ML_KIT_CHINESE.usesMlKit());
        assertNull(ModelChoice.Ocr.ML_KIT_CHINESE.recognitionFile);
        assertNull(ModelChoice.Ocr.ML_KIT_CHINESE.dictionaryFile);
        for (ModelChoice.Ocr ocr : ModelChoice.Ocr.values()) {
            if (!ocr.usesMlKit() && ocr != ModelChoice.Ocr.DBNET_CRNN) assertEquals(48, ocr.recognitionHeight());
        }
    }
    @Test public void storedModelIdsRestoreEveryOptionAndMigrateLegacyPreferences() {
        for (ModelChoice.Ocr ocr : ModelChoice.Ocr.values()) {
            assertEquals(ocr, ModelChoice.Ocr.fromStoredId(ocr.name(), false));
            assertEquals(ocr, ModelChoice.Ocr.fromStoredId(ocr.name(), true));
        }
        assertEquals(ModelChoice.Ocr.V4, ModelChoice.Ocr.fromStoredId(null, false));
        assertEquals(ModelChoice.Ocr.V6_TINY, ModelChoice.Ocr.fromStoredId(null, true));
        assertEquals(ModelChoice.Ocr.V4, ModelChoice.Ocr.fromStoredId("REMOVED_MODEL", false));
        assertEquals(ModelChoice.Ocr.V6_TINY, ModelChoice.Ocr.fromStoredId("", true));
    }
    @Test public void bothYoloOutputSizesMapToSameOriginalCoordinates() {
        for (int side : new int[]{640, 1280}) {
            int count = side == 640 ? 8400 : 33600;
            FloatBuffer buffer = FloatBuffer.allocate(5 * count);
            double scale = side / 1000.0;
            int padY = side / 4;
            buffer.put(0, (float)(500 * scale));
            buffer.put(count, (float)(250 * scale + padY));
            buffer.put(2 * count, (float)(100 * scale));
            buffer.put(3 * count, (float)(100 * scale));
            buffer.put(4 * count, .9f);
            List<Detection> result = TensorDecoders.yolo(buffer, 5, count, 1000, 500, scale, 0, padY, .25, .7);
            assertEquals(1, result.size());
            assertArrayEquals(new double[]{450,200,550,300}, result.get(0).box, .001);
        }
    }
}
