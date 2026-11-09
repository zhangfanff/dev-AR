package com.example.cabinetmatcher.camera;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;

/** Uses the same upright analysis-image coordinates as the cabinet detections. */
final class HandDrawing {
    static void draw(Canvas canvas,float[][] hands,int width,int height,Matrix matrix,float unit) {
        if(hands.length==0) return;
        RectF content=new RectF(0,0,width,height); matrix.mapRect(content);
        int saved=canvas.save(); canvas.clipRect(content);
        Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);
        for(int h=0;h<hands.length;h++) {
            float[] xy=HandSkeleton.pixels(hands[h],width,height); matrix.mapPoints(xy);
            int color=h==0 ? 0xff32d6ff : 0xffff82d9;
            paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(2.5f*unit); paint.setColor(color);
            for(int[] edge:HandSkeleton.EDGES) canvas.drawLine(xy[2*edge[0]],xy[2*edge[0]+1],xy[2*edge[1]],xy[2*edge[1]+1],paint);
            paint.setStyle(Paint.Style.FILL);
            for(int i=0;i<21;i++) {
                paint.setColor(i==HandSkeleton.INDEX_TIP ? 0xffffdf55 : color);
                canvas.drawCircle(xy[2*i],xy[2*i+1],(i==HandSkeleton.INDEX_TIP ? 5 : 3.5f)*unit,paint);
            }
        }
        canvas.restoreToCount(saved);
    }
}
