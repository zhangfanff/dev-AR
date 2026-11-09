package com.example.cabinetmatcher.camera;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import com.example.cabinetmatcher.camera.core.Detection;
import com.example.cabinetmatcher.camera.core.MatchReport;
import java.util.HashMap;
import java.util.Map;

final class Annotations {
    static final int[] COLORS = {Color.rgb(0, 143, 83), Color.rgb(9, 105, 218), Color.rgb(130, 80, 223), Color.rgb(181, 71, 8), Color.rgb(207, 34, 46)};
    static Bitmap draw(Bitmap original, MatchReport report) {
        Bitmap bitmap = original.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setStrokeWidth(Math.max(2, bitmap.getWidth() / 550f));
        paint.setTextSize(Math.max(20, bitmap.getWidth() / 65f));
        Map<Integer, Integer> knobColors = new HashMap<>(), textColors = new HashMap<>();
        Map<Integer, Integer> textOwners = new HashMap<>();
        for (int i = 0; i < report.matches.size(); i++) {
            MatchReport.Pair pair = report.matches.get(i);
            int color = "recognized".equals(pair.text.recognitionStatus) ? COLORS[i % COLORS.length] : Color.rgb(227, 98, 9);
            knobColors.put(pair.knob.id, color); textColors.put(pair.text.id, color);
            textOwners.put(pair.text.id,pair.knob.id);
        }
        for (Detection text : report.texts) box(canvas, paint, text, textColors.getOrDefault(text.id, Color.GRAY),
                "R" + text.id + (textOwners.containsKey(text.id) ? " ← K"+textOwners.get(text.id)+" · "+RecognitionLabel.shortStatus(text) : " 未选中"));
        for (Detection knob : report.knobs) box(canvas, paint, knob, knobColors.getOrDefault(knob.id, Color.rgb(227, 98, 9)), "K" + knob.id);
        for (MatchReport.Pair pair : report.matches) {
            paint.setColor(knobColors.get(pair.knob.id)); paint.setStyle(Paint.Style.STROKE);
            canvas.drawLine((float) pair.knob.cx, (float) pair.knob.cy, (float) pair.text.cx, (float) pair.text.cy, paint);
            double angle = Math.atan2(pair.text.cy - pair.knob.cy, pair.text.cx - pair.knob.cx), size = Math.max(12, bitmap.getWidth() / 90f);
            Path arrow = new Path(); arrow.moveTo((float) pair.text.cx, (float) pair.text.cy);
            for (double delta : new double[]{-0.45, 0.45}) arrow.lineTo((float) (pair.text.cx - size * Math.cos(angle + delta)), (float) (pair.text.cy - size * Math.sin(angle + delta)));
            arrow.close(); paint.setStyle(Paint.Style.FILL); canvas.drawPath(arrow, paint);
        }
        return bitmap;
    }
    private static void box(Canvas canvas, Paint paint, Detection item, int color, String label) {
        paint.setColor(color); paint.setStyle(Paint.Style.STROKE);
        if (item.polygon != null) {
            Path path = new Path(); path.moveTo((float) item.polygon[0][0], (float) item.polygon[0][1]);
            for (int i = 1; i < item.polygon.length; i++) path.lineTo((float) item.polygon[i][0], (float) item.polygon[i][1]);
            path.close(); canvas.drawPath(path, paint);
        } else canvas.drawRect(new RectF((float) item.box[0], (float) item.box[1], (float) item.box[2], (float) item.box[3]), paint);
        float x = (float) item.box[0], y = Math.max(paint.getTextSize(), (float) item.box[1] - 5);
        paint.setStyle(Paint.Style.FILL); paint.setColor(Color.WHITE);
        canvas.drawRect(x, y - paint.getTextSize(), x + paint.measureText(label) + 8, y + 4, paint);
        paint.setColor(color); canvas.drawText(label, x + 4, y, paint);
    }
}
