package com.example.cabinetmatcher.camera;

final class HandSkeleton {
    static final int LANDMARKS = 21, INDEX_TIP = 8;
    static final int[][] EDGES = {
        {0,1},{1,2},{2,3},{3,4}, {0,5},{5,6},{6,7},{7,8},
        {5,9},{9,10},{10,11},{11,12}, {9,13},{13,14},{14,15},{15,16},
        {13,17},{17,18},{18,19},{19,20},{0,17}
    };
    static float[] pixels(float[] xyz, int width, int height) {
        if (xyz.length != LANDMARKS * 3 || width <= 0 || height <= 0)
            throw new IllegalArgumentException("Invalid landmark dimensions");
        float[] xy = new float[LANDMARKS * 2];
        for (int i=0;i<LANDMARKS;i++) { xy[2*i]=xyz[3*i]*width; xy[2*i+1]=xyz[3*i+1]*height; }
        return xy;
    }
}
