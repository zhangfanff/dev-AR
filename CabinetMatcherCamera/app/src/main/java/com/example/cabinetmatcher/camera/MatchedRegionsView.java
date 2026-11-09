package com.example.cabinetmatcher.camera;

import android.content.Context;
import android.graphics.*;
import android.text.TextUtils;
import android.text.TextPaint;
import android.view.View;
import com.example.cabinetmatcher.camera.core.*;
import java.util.*;

/** Draw raw assigned regions from the same frame, regardless of the OCR outcome.
 * Borrows the FrameResult bitmap; does not allocate one crop bitmap per region. */
final class MatchedRegionsView extends View {
    private final Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG|Paint.FILTER_BITMAP_FLAG);
    private final TextPaint text=new TextPaint(Paint.ANTI_ALIAS_FLAG);
    private final float density;
    private MatchReport report;
    private Bitmap source;
    private double threshold;
    private final Map<Integer,MatchReport.Pair> pairs=new HashMap<>();
    private final Map<Integer,MatchReport.Decision> decisions=new HashMap<>();

    MatchedRegionsView(Context context) {
        super(context);density=getResources().getDisplayMetrics().density;
        text.setTextSize(12*getResources().getDisplayMetrics().scaledDensity);
        setBackgroundColor(Color.WHITE);
    }
    void show(MatchReport report,Bitmap source,double threshold) {
        this.report=report;this.source=source;this.threshold=threshold;
        pairs.clear();decisions.clear();
        if(report!=null) {
            for(MatchReport.Pair pair:report.matches)pairs.put(pair.knob.id,pair);
            for(MatchReport.Decision decision:report.decisions)decisions.put(decision.knob.id,decision);
        }
        requestLayout();invalidate();
    }
    @Override protected void onMeasure(int widthSpec,int heightSpec) {
        int rows=report==null?0:report.knobs.size();
        int height=Math.round((rows==0?50:rows*216+8)*density);
        setMeasuredDimension(resolveSize(Math.round(320*density),widthSpec),resolveSize(height,heightSpec));
    }
    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if(report==null||report.knobs.isEmpty()) {
            line(canvas,report==null?"等待完成帧：匹配失败的 OCR 区域也会显示在这里。":"这一帧没有检测到旋钮。",8,27,0xff455a64);
            return;
        }
        for(int i=0;i<report.knobs.size();i++) {
            Detection knob=report.knobs.get(i);MatchReport.Pair pair=pairs.get(knob.id);
            float top=(i*216+4)*density;
            int saved=canvas.save();canvas.translate(0,top);
            if(pair==null)drawUnmatched(canvas,knob,decisions.get(knob.id));
            else drawPair(canvas,pair);
            paint.setStyle(Paint.Style.STROKE);paint.setColor(0xffcbd5e1);paint.setStrokeWidth(density);
            canvas.drawLine(0,211*density,getWidth(),211*density,paint);canvas.restoreToCount(saved);
        }
    }
    private void drawPair(Canvas canvas,MatchReport.Pair pair) {
        Detection region=pair.text;
        int color="recognized".equals(region.recognitionStatus)?0xff008f53:0xffb45309;
        line(canvas,"K"+pair.knob.id+" → R"+region.id+" · 已选中文本框",8,19,color);
        String status=RecognitionLabel.describe(region);
        if("low_confidence".equals(region.recognitionStatus))status=String.format(Locale.CHINA,"OCR 置信度 %.3f < 阈值 %.3f",region.confidence,threshold);
        line(canvas,status,8,40,color);
        line(canvas,"OCR 原始输出："+(region.text==null||region.text.isEmpty()?"（空）":region.text),8,61,0xff102a43);
        line(canvas,String.format(Locale.CHINA,"距离 %.1f / %.1fpx · 框外 %.1f%% · 夹角 %.1f°",
                pair.distance,report.maxDistanceFor(pair.knob),100*TextOverlap.outsideRatio(pair.knob,region),
                report.searchDirection.angle(region.cx-pair.knob.cx,region.cy-pair.knob.cy)),8,82,0xff455a64);
        RectF area=new RectF(8*density,91*density,getWidth()-8*density,187*density);
        paint.setStyle(Paint.Style.FILL);paint.setColor(0xffe8eef3);canvas.drawRect(area,paint);
        if(source!=null&&!source.isRecycled()) {
            double sx=(double)source.getWidth()/report.width,sy=(double)source.getHeight()/report.height;
            int left=Math.max(0,Math.min(source.getWidth()-1,(int)Math.floor(region.box[0]*sx)));
            int top=Math.max(0,Math.min(source.getHeight()-1,(int)Math.floor(region.box[1]*sy)));
            int right=Math.max(left+1,Math.min(source.getWidth(),(int)Math.ceil(region.box[2]*sx)));
            int bottom=Math.max(top+1,Math.min(source.getHeight(),(int)Math.ceil(region.box[3]*sy)));
            Rect crop=new Rect(left,top,right,bottom);
            float scale=Math.min(area.width()/crop.width(),area.height()/crop.height());
            float width=crop.width()*scale,height=crop.height()*scale;
            RectF destination=new RectF(area.centerX()-width/2,area.centerY()-height/2,area.centerX()+width/2,area.centerY()+height/2);
            canvas.drawBitmap(source,crop,destination,paint);
            if(region.polygon!=null) {
                Path path=new Path();
                for(int i=0;i<region.polygon.length;i++) {
                    float x=destination.left+(float)(region.polygon[i][0]*sx-left)*scale;
                    float y=destination.top+(float)(region.polygon[i][1]*sy-top)*scale;
                    if(i==0)path.moveTo(x,y);else path.lineTo(x,y);
                }
                path.close();paint.setStyle(Paint.Style.STROKE);paint.setColor(color);paint.setStrokeWidth(density);
                canvas.drawPath(path,paint);
            }
        } else line(canvas,"原图区域不可用",12,142,0xff455a64);
        line(canvas,"原图放大预览；轮廓为实际文字框，放大不改变识别输入。",8,204,0xff455a64);
    }
    private void drawUnmatched(Canvas canvas,Detection knob,MatchReport.Decision decision) {
        line(canvas,"K"+knob.id+" · 没有几何配对（未选中文本框）",8,19,0xffb45309);
        line(canvas,decision==null?"请查看该帧 JSON 候选信息。":GreedyMatcher.describe(decision.reason),8,43,0xff455a64);
        if(decision==null)return;
        line(canvas,String.format(Locale.CHINA,"距离上限 %.1fpx%s",report.maxDistanceFor(knob),report.knobRelative?"；文本框外面积 ≥ 33.33%":""),8,65,0xff455a64);
        List<MatchReport.Candidate> candidates=new ArrayList<>(decision.candidates);
        candidates.sort(Comparator.comparingDouble(candidate->candidate.distance));
        for(int i=0;i<Math.min(3,candidates.size());i++) {
            MatchReport.Candidate candidate=candidates.get(i);
            StringBuilder reasons=new StringBuilder();
            for(String reason:candidate.reasons) {
                if(reasons.length()>0)reasons.append("；");reasons.append(GreedyMatcher.describe(reason));
            }
            line(canvas,"R"+candidate.text.id+String.format(Locale.CHINA," %.1fpx：",candidate.distance)+reasons,8,92+i*30,0xff455a64);
        }
        line(canvas,"灰框是未选中的检测区域；全部候选及原因见 JSON。",8,204,0xff455a64);
    }
    private void line(Canvas canvas,String value,float x,float baseline,int color) {
        text.setColor(color);
        CharSequence fitted=TextUtils.ellipsize(value,text,Math.max(1,getWidth()-x*density-8*density),TextUtils.TruncateAt.END);
        canvas.drawText(fitted.toString(),x*density,baseline*density,text);
    }
}
