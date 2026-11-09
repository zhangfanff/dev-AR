package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import java.nio.ByteBuffer;
import static org.junit.Assert.*;

public class YuvPackingTest {
    @Test public void planarPaddingIsExcludedAndChromaUsesVUOrder() {
        ByteBuffer y = ByteBuffer.wrap(new byte[]{1,2,3,4,99,99,5,6,7,8,99,99,9,10,11,12,99,99,13,14,15,16});
        ByteBuffer u = ByteBuffer.wrap(new byte[]{21,22,99,23,24});
        ByteBuffer v = ByteBuffer.wrap(new byte[]{31,32,99,33,34});
        assertArrayEquals(new byte[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,21,32,22,33,23,34,24},
                YuvPacking.nv21(4,4,y,6,1,u,3,1,v,3,1));
    }
    @Test public void interleavedChromaWithOffsetsAndRowPaddingWorks() {
        ByteBuffer y = ByteBuffer.wrap(new byte[]{99,1,2,3,4,99,99,5,6,7,8}); y.position(1);
        ByteBuffer interleaved = ByteBuffer.wrap(new byte[]{31,21,32,22});
        ByteBuffer u = interleaved.duplicate(); u.position(1); ByteBuffer v = interleaved.duplicate();
        assertArrayEquals(new byte[]{1,2,3,4,5,6,7,8,31,21,32,22},
                YuvPacking.nv21(4,2,y,6,1,u,4,2,v,4,2));
        assertEquals(1, y.position()); assertEquals(1, u.position()); assertEquals(0, v.position());
    }
    @Test public void readOnlyPlanesCanBePacked() {
        ByteBuffer y = ByteBuffer.wrap(new byte[]{1,2,3,4}).asReadOnlyBuffer();
        assertArrayEquals(new byte[]{1,2,3,4,6,5}, YuvPacking.nv21(2,2,y,2,1,
                ByteBuffer.wrap(new byte[]{5}).asReadOnlyBuffer(),1,1,
                ByteBuffer.wrap(new byte[]{6}).asReadOnlyBuffer(),1,1));
    }
    @Test(expected=IllegalArgumentException.class) public void oddFrameSizeIsRejected() {
        YuvPacking.nv21(3,2,ByteBuffer.allocate(10),3,1,ByteBuffer.allocate(3),2,1,ByteBuffer.allocate(3),2,1);
    }
}
