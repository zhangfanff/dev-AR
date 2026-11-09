package com.example.cabinetmatcher.camera.core;

import java.util.ArrayList;
import java.util.List;

/** Polygon area outside one axis-aligned YOLO box. No native runtime required. */
public final class TextOverlap {
    private TextOverlap() {}
    public static double outsideRatio(Detection knob,Detection text) {
        double[][] shape=text.polygon!=null?text.polygon:new double[][]{
                {text.box[0],text.box[1]},{text.box[2],text.box[1]},
                {text.box[2],text.box[3]},{text.box[0],text.box[3]}};
        List<double[]> polygon=new ArrayList<>();
        for(double[] p:shape) {
            if(p==null||p.length<2||!Double.isFinite(p[0])||!Double.isFinite(p[1]))return 0;
            polygon.add(p);
        }
        double total=area(polygon);if(!(total>0))return 0;
        polygon=clip(polygon,0,knob.box[0],true);
        polygon=clip(polygon,0,knob.box[2],false);
        polygon=clip(polygon,1,knob.box[1],true);
        polygon=clip(polygon,1,knob.box[3],false);
        return Math.max(0,Math.min(1,1-area(polygon)/total));
    }
    private static List<double[]> clip(List<double[]> input,int axis,double boundary,boolean greater) {
        List<double[]> output=new ArrayList<>();if(input.isEmpty())return output;
        double[] previous=input.get(input.size()-1);
        boolean previousInside=greater?previous[axis]>=boundary:previous[axis]<=boundary;
        for(double[] current:input) {
            boolean inside=greater?current[axis]>=boundary:current[axis]<=boundary;
            if(inside!=previousInside) {
                double fraction=(boundary-previous[axis])/(current[axis]-previous[axis]);
                output.add(new double[]{previous[0]+fraction*(current[0]-previous[0]),previous[1]+fraction*(current[1]-previous[1])});
            }
            if(inside)output.add(current);
            previous=current;previousInside=inside;
        }
        return output;
    }
    private static double area(List<double[]> polygon) {
        if(polygon.size()<3)return 0;
        double x=polygon.get(0)[0],y=polygon.get(0)[1],twice=0;
        for(int i=1;i+1<polygon.size();i++) {
            double[] a=polygon.get(i),b=polygon.get(i+1);
            twice+=(a[0]-x)*(b[1]-y)-(b[0]-x)*(a[1]-y);
        }
        return Math.abs(twice)*.5;
    }
}
