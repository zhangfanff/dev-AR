package com.example.cabinetmatcher.camera.core;

/** Directions are in the upright analysis frame: x right, y down. ALL is a union, never four assignments. */
public enum SearchDirection {
    DOWN("下方",0,1), UP("上方",0,-1), LEFT("左侧",-1,0), RIGHT("右侧",1,0), ALL("四向同时",0,0);
    public final String label;
    private final int x,y;
    SearchDirection(String label,int x,int y) { this.label=label; this.x=x; this.y=y; }
    public SearchDirection axis(double dx,double dy) {
        if(this!=ALL) return this;
        return Math.abs(dy)>=Math.abs(dx) ? (dy>=0 ? DOWN : UP) : (dx>=0 ? RIGHT : LEFT);
    }
    public boolean ahead(double dx,double dy) { SearchDirection axis=axis(dx,dy); return dx*axis.x+dy*axis.y>0; }
    public double lateral(double dx,double dy) { SearchDirection axis=axis(dx,dy); return Math.abs(dx*axis.y-dy*axis.x); }
    public double angle(double dx,double dy) {
        SearchDirection axis=axis(dx,dy);
        return Math.toDegrees(Math.atan2(lateral(dx,dy),dx*axis.x+dy*axis.y));
    }
}
