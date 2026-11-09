package com.example.cabinetmatcher.camera.core;

import java.nio.ByteBuffer;

/** Converts planar or interleaved YUV_420_888 planes, including row padding, to NV21. */
public final class YuvPacking {
    private YuvPacking() {}
    public static byte[] nv21(int width, int height, ByteBuffer y, int yRow, int yPixel,
                              ByteBuffer u, int uRow, int uPixel, ByteBuffer v, int vRow, int vPixel) {
        if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0)
            throw new IllegalArgumentException("YUV frame dimensions must be positive and even");
        byte[] output = new byte[Math.multiplyExact(width, height) * 3 / 2];
        copyPlane(y, yRow, yPixel, width, height, output, 0, 1);
        int offset = width * height;
        copyPlane(v, vRow, vPixel, width / 2, height / 2, output, offset, 2);
        copyPlane(u, uRow, uPixel, width / 2, height / 2, output, offset + 1, 2);
        return output;
    }
    private static void copyPlane(ByteBuffer plane, int rowStride, int pixelStride, int width, int height,
                                  byte[] output, int start, int step) {
        if (rowStride <= 0 || pixelStride <= 0) throw new IllegalArgumentException("Invalid YUV strides");
        int base = plane.position();
        for (int row = 0; row < height; row++) {
            int source = base + row * rowStride;
            int destination = start + row * width * step;
            for (int column = 0; column < width; column++)
                output[destination + column * step] = plane.get(source + column * pixelStride);
        }
    }
}
