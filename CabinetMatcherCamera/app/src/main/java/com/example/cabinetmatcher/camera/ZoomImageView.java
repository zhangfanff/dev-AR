package com.example.cabinetmatcher.camera;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;

final class ZoomImageView extends View {
    private Bitmap bitmap;
    private float scale = 1, minScale = 1, offsetX, offsetY, previousX, previousY;
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final ScaleGestureDetector detector;
    ZoomImageView(Context context) {
        super(context); setBackgroundColor(Color.rgb(225, 233, 240));
        detector = new ScaleGestureDetector(context, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override public boolean onScale(ScaleGestureDetector gesture) {
                float next = Math.max(minScale, Math.min(minScale * 8, scale * gesture.getScaleFactor()));
                float factor = next / scale;
                offsetX = gesture.getFocusX() - (gesture.getFocusX() - offsetX) * factor;
                offsetY = gesture.getFocusY() - (gesture.getFocusY() - offsetY) * factor;
                scale = next; clamp(); invalidate(); return true;
            }
        });
        setContentDescription("图片预览，可双指缩放并拖动查看旋钮与文本连线");
    }
    void show(Bitmap bitmap, boolean reset) { this.bitmap = bitmap; if (reset) fit(); invalidate(); }
    private void fit() {
        if (bitmap == null || getWidth() == 0 || getHeight() == 0) return;
        minScale = Math.min((float) getWidth() / bitmap.getWidth(), (float) getHeight() / bitmap.getHeight());
        scale = minScale; offsetX = (getWidth() - bitmap.getWidth() * scale) / 2; offsetY = (getHeight() - bitmap.getHeight() * scale) / 2;
    }
    private void clamp() {
        if (bitmap == null) return;
        float w = bitmap.getWidth() * scale, h = bitmap.getHeight() * scale;
        offsetX = w <= getWidth() ? (getWidth() - w) / 2 : Math.max(getWidth() - w, Math.min(0, offsetX));
        offsetY = h <= getHeight() ? (getHeight() - h) / 2 : Math.max(getHeight() - h, Math.min(0, offsetY));
    }
    @Override protected void onSizeChanged(int w, int h, int oldw, int oldh) { fit(); }
    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (bitmap != null && !bitmap.isRecycled()) { canvas.save(); canvas.translate(offsetX, offsetY); canvas.scale(scale, scale); canvas.drawBitmap(bitmap, 0, 0, paint); canvas.restore(); }
        else { paint.setColor(Color.rgb(85, 108, 130)); paint.setTextSize(18 * getResources().getDisplayMetrics().scaledDensity); canvas.drawText("等待视频帧处理结果", 24, getHeight() / 2f, paint); }
    }
    @Override public boolean onTouchEvent(MotionEvent event) {
        if (bitmap == null) return super.onTouchEvent(event);
        detector.onTouchEvent(event);
        if (event.getActionMasked() == MotionEvent.ACTION_DOWN) { previousX = event.getX(); previousY = event.getY(); getParent().requestDisallowInterceptTouchEvent(true); return true; }
        if (event.getActionMasked() == MotionEvent.ACTION_MOVE) {
            if (!detector.isInProgress() && event.getPointerCount() == 1 && scale > minScale) {
                offsetX += event.getX() - previousX; offsetY += event.getY() - previousY; clamp(); invalidate();
            }
            previousX = event.getX(); previousY = event.getY(); return true;
        }
        if (event.getActionMasked() == MotionEvent.ACTION_UP) { getParent().requestDisallowInterceptTouchEvent(false); performClick(); return true; }
        if (event.getActionMasked() == MotionEvent.ACTION_CANCEL) getParent().requestDisallowInterceptTouchEvent(false);
        return true;
    }
    @Override public boolean performClick() { super.performClick(); return true; }
}
