package com.example.cabinetmatcher.camera;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.text.TextUtils;
import android.text.TextPaint;
import android.view.View;
import androidx.annotation.OptIn;
import androidx.camera.view.PreviewView;
import androidx.camera.view.TransformExperimental;
import androidx.camera.view.transform.CoordinateTransform;
import androidx.camera.view.transform.OutputTransform;
import com.example.cabinetmatcher.camera.core.Detection;
import com.example.cabinetmatcher.camera.core.MatchReport;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/** Latest completed detections over the live stream. Intentionally no tracking/latency correction. */
@OptIn(markerClass = TransformExperimental.class)
final class LiveOverlayView extends View {
    private static final int[] COLORS = {0xff38ef9a, 0xff52d6ff, 0xffffdf55, 0xffff7fd4, 0xffbda0ff};
    interface PreviewTransform { OutputTransform getOutputTransform(); }
    private final PreviewTransform preview;
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final TextPaint text = new TextPaint(Paint.ANTI_ALIAS_FLAG);
    private final float density;
    private MainActivity.FrameResult frame;
    private String state = "等待开始识别";
    private double updatesPerSecond;
    private boolean ticking;
    private final Runnable ticker = new Runnable() {
        @Override public void run() { if (ticking) { invalidate(); postDelayed(this, 100); } }
    };

    LiveOverlayView(Context context, PreviewView preview) {
        this(context, preview::getOutputTransform);
    }
    LiveOverlayView(Context context, PreviewTransform preview) {
        super(context); this.preview = preview; density = getResources().getDisplayMetrics().density;
        text.setTextSize(12 * getResources().getDisplayMetrics().scaledDensity);
        setClickable(false); setFocusable(false); setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_NO);
    }
    void show(MainActivity.FrameResult frame, double updatesPerSecond) {
        this.frame = frame; this.updatesPerSecond = updatesPerSecond; invalidate();
    }
    void clear() { frame = null; updatesPerSecond = 0; invalidate(); }
    void state(String state) { this.state = state; invalidate(); }
    @Override protected void onAttachedToWindow() { super.onAttachedToWindow(); ticking = true; post(ticker); }
    @Override protected void onDetachedFromWindow() { ticking = false; removeCallbacks(ticker); super.onDetachedFromWindow(); }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        MainActivity.FrameResult current = frame;
        String mapping = "";
        if (current != null) {
            OutputTransform target = preview.getOutputTransform();
            if (target != null && current.analysisTransform != null) {
                Matrix matrix = new Matrix();
                new CoordinateTransform(current.analysisTransform, target).transform(matrix);
                drawMatches(canvas, current.report, matrix);
                HandDrawing.draw(canvas, current.hands, current.report.width, current.report.height, matrix, density);
            } else mapping = " · 等待预览坐标";
        }
        drawHud(canvas, current, mapping);
    }
    private void drawMatches(Canvas canvas, MatchReport report, Matrix matrix) {
        RectF content = new RectF(0, 0, report.width, report.height); matrix.mapRect(content);
        int saved = canvas.save(); canvas.clipRect(content);
        Set<Integer> matched = new HashSet<>();
        Set<Integer> assignedRegions = new HashSet<>();
        for (MatchReport.Pair pair : report.matches) assignedRegions.add(pair.text.id);
        for (Detection region : report.texts) if (!assignedRegions.contains(region.id)) {
            RectF bounds = outline(canvas, region, matrix, 0xffaebcc8);
            label(canvas, "R"+region.id+" 未选中", bounds.left, bounds.top, 0xffaebcc8);
        }
        for (int i = 0; i < report.matches.size(); i++) {
            MatchReport.Pair pair = report.matches.get(i); matched.add(pair.knob.id);
            int color = "recognized".equals(pair.text.recognitionStatus) ? COLORS[i % COLORS.length] : 0xffffad42;
            RectF knobBounds = outline(canvas, pair.knob, matrix, color);
            label(canvas,"K"+pair.knob.id+" → R"+pair.text.id,knobBounds.left,knobBounds.top,color);
            RectF textBounds = outline(canvas, pair.text, matrix, color);
            float[] centers = {(float)pair.knob.cx, (float)pair.knob.cy, (float)pair.text.cx, (float)pair.text.cy};
            matrix.mapPoints(centers); paint.setColor(color); paint.setStrokeWidth(2 * density); paint.setStyle(Paint.Style.STROKE);
            canvas.drawLine(centers[0], centers[1], centers[2], centers[3], paint);
            double angle = Math.atan2(centers[3]-centers[1], centers[2]-centers[0]);
            float arrow = 9 * density;
            for (double delta : new double[]{-.45, .45}) canvas.drawLine(centers[2], centers[3],
                    centers[2]-(float)(arrow*Math.cos(angle+delta)), centers[3]-(float)(arrow*Math.sin(angle+delta)), paint);
            String value = pair.text.text == null || pair.text.text.isEmpty() ? "（未读出）" : pair.text.text;
            label(canvas, "K"+pair.knob.id+" → R"+pair.text.id+" · "+RecognitionLabel.shortStatus(pair.text)+"  "+value, textBounds.left, textBounds.top, color);
        }
        for (Detection knob : report.knobs) if (!matched.contains(knob.id)) {
            RectF bounds = outline(canvas, knob, matrix, 0xffffad42);
            label(canvas, "K"+knob.id+" 未匹配", bounds.left, bounds.top, 0xffffad42);
        }
        canvas.restoreToCount(saved);
    }
    private RectF outline(Canvas canvas, Detection item, Matrix matrix, int color) {
        double[][] polygon = item.polygon != null ? item.polygon : new double[][]{
                {item.box[0],item.box[1]}, {item.box[2],item.box[1]}, {item.box[2],item.box[3]}, {item.box[0],item.box[3]}};
        float[] points = new float[polygon.length*2];
        for (int i=0; i<polygon.length; i++) { points[2*i]=(float)polygon[i][0]; points[2*i+1]=(float)polygon[i][1]; }
        matrix.mapPoints(points); Path path = new Path(); path.moveTo(points[0],points[1]);
        RectF bounds = new RectF(points[0],points[1],points[0],points[1]);
        for (int i=0; i<polygon.length; i++) {
            if (i>0) path.lineTo(points[2*i],points[2*i+1]);
            bounds.left=Math.min(bounds.left,points[2*i]); bounds.right=Math.max(bounds.right,points[2*i]);
            bounds.top=Math.min(bounds.top,points[2*i+1]); bounds.bottom=Math.max(bounds.bottom,points[2*i+1]);
        }
        path.close(); paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(2*density); paint.setColor(color);
        canvas.drawPath(path,paint); return bounds;
    }
    private void label(Canvas canvas, String value, float x, float y, int color) {
        float pad=3*density, margin=4*density;
        CharSequence fitted=TextUtils.ellipsize(value,text,Math.max(1,getWidth()-2*margin-2*pad),TextUtils.TruncateAt.END);
        String line=fitted.toString(); float width=text.measureText(line)+2*pad;
        x=Math.max(margin,Math.min(x,getWidth()-width-margin));
        float baseline=Math.max(text.getTextSize()+pad,Math.min(y-3*density,getHeight()-pad));
        paint.setStyle(Paint.Style.FILL); paint.setColor(0xdd071725);
        canvas.drawRect(x,baseline-text.getTextSize()-pad,x+width,baseline+pad,paint);
        text.setColor(color); canvas.drawText(line,x+pad,baseline,text);
    }
    private void drawHud(Canvas canvas, MainActivity.FrameResult current, String mapping) {
        String[] lines;
        if (current == null) lines = new String[]{state, "等待当前会话的匹配结果"};
        else lines = new String[]{state+mapping,
                String.format(Locale.CHINA,"帧 #%d · 结果年龄 %.1fs · 更新 %.2f 帧/s", current.ticket.sequence,
                        (System.nanoTime()-current.ticket.receivedNs)/1e9,updatesPerSecond),
                String.format(Locale.CHINA,"处理 %.0fms · 匹配 %d · 手 %d · 方向 %s", (current.finishedNs-current.ticket.receivedNs)/1e6,
                        current.report.matches.size(),current.hands.length,current.settings.searchDirection.label),
                (current.settings.inferenceMode == com.example.cabinetmatcher.camera.core.InferenceMode.PARALLEL ? "并行" : "串行")
                        + " · " + current.settings.models.yoloSide+" / "+current.settings.models.ocr.label+" · 同一帧结果"};
        float pad=7*density, lineHeight=text.getTextSize()*1.4f;
        paint.setStyle(Paint.Style.FILL); paint.setColor(0xd9071725);
        canvas.drawRect(0,0,getWidth(),pad*2+lines.length*lineHeight,paint); text.setColor(Color.WHITE);
        for (int i=0;i<lines.length;i++) {
            CharSequence fitted=TextUtils.ellipsize(lines[i],text,Math.max(1,getWidth()-pad*2),TextUtils.TruncateAt.END);
            canvas.drawText(fitted.toString(),pad,pad+text.getTextSize()+i*lineHeight,text);
        }
    }
    boolean hasResultForTest() { return frame != null; }
}
