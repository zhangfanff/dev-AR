package com.example.cabinetmatcher.camera.core;

import java.util.ArrayList;
import java.util.List;

public final class MatchReport {
    public final int width, height;
    public final List<Detection> knobs, texts;
    public final List<Pair> matches = new ArrayList<>();
    public final List<Decision> decisions = new ArrayList<>();
    public final List<Detection> unmatchedTexts = new ArrayList<>();
    public final double maxDistance, maxHorizontal;
    public final double maxAngleFromDownwardDegrees;
    public final SearchDirection searchDirection;
    public final boolean knobRelative;
    public static final double MIN_TEXT_OUTSIDE_RATIO=1.0/3.0;
    public double maxDistanceFor(Detection knob) {
        return knobRelative?Math.hypot(knob.box[2]-knob.box[0],knob.box[3]-knob.box[1]):maxDistance;
    }
    public long elapsedMs;
    public TimingReport timing = new TimingReport();
    public String imageName;
    public int sourceWidth, sourceHeight;
    public int recognizedMatches() {
        int count = 0;
        for (Pair pair : matches) if ("recognized".equals(pair.text.recognitionStatus)) count++;
        return count;
    }
    public MatchReport(int width, int height, List<Detection> knobs, List<Detection> texts,
                       double maxDistance, double maxHorizontal) {
        this(width, height, knobs, texts, maxDistance, maxHorizontal, 45);
    }
    public MatchReport(int width, int height, List<Detection> knobs, List<Detection> texts,
                       double maxDistance, double maxHorizontal, double maxAngleFromDownwardDegrees) {
        this(width,height,knobs,texts,maxDistance,maxHorizontal,maxAngleFromDownwardDegrees,SearchDirection.DOWN);
    }
    public MatchReport(int width, int height, List<Detection> knobs, List<Detection> texts,
                       double maxDistance, double maxHorizontal, double maxAngleFromDownwardDegrees, SearchDirection searchDirection) {
        this(width,height,knobs,texts,maxDistance,maxHorizontal,maxAngleFromDownwardDegrees,searchDirection,false);
    }
    public MatchReport(int width,int height,List<Detection> knobs,List<Detection> texts,
                       double maxDistance,double maxHorizontal,double angle,SearchDirection direction,boolean knobRelative) {
        this.width = width; this.height = height; this.knobs = knobs; this.texts = texts;
        this.sourceWidth = width; this.sourceHeight = height;
        this.maxDistance = knobRelative?Double.NaN:maxDistance; this.maxHorizontal = maxHorizontal;
        this.maxAngleFromDownwardDegrees = angle;
        this.searchDirection = direction;this.knobRelative=knobRelative;
    }
    public static final class Pair {
        public final Detection knob, text;
        public final int rank;
        public final double distance;
        public Pair(Detection knob, Detection text, int rank, double distance) {
            this.knob = knob; this.text = text; this.rank = rank; this.distance = distance;
        }
    }
    public static final class Candidate {
        public final Detection text;
        public final double dx, dy, distance;
        public final double maxDistance,outsideRatio;
        public final boolean geometric;
        public final Integer owner;
        public final List<String> reasons;
        public boolean selected;
        public Candidate(Detection text, double dx, double dy, double distance,
                         boolean geometric, Integer owner, List<String> reasons) {
            this(text,dx,dy,distance,geometric,owner,reasons,Double.NaN,Double.NaN);
        }
        public Candidate(Detection text,double dx,double dy,double distance,boolean geometric,Integer owner,
                         List<String> reasons,double maxDistance,double outsideRatio) {
            this.text = text; this.dx = dx; this.dy = dy; this.distance = distance;
            this.geometric = geometric; this.owner = owner; this.reasons = reasons;
            this.maxDistance=maxDistance;this.outsideRatio=outsideRatio;
        }
    }
    public static final class Decision {
        public final Detection knob;
        public final int rank;
        public final List<Candidate> candidates = new ArrayList<>();
        public Pair pair;
        public String reason;
        public Decision(Detection knob, int rank) { this.knob = knob; this.rank = rank; }
    }
}
