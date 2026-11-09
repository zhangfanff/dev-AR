package com.example.cabinetmatcher.camera.core;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class GreedyMatcher {
    private GreedyMatcher() {}
    public static MatchReport match(int width, int height, List<Detection> knobs,
                                     List<Detection> texts, double maxDistance, double maxHorizontal) {
        return match(width, height, knobs, texts, maxDistance, maxHorizontal, 45);
    }
    public static MatchReport match(int width, int height, List<Detection> knobs,
                                     List<Detection> texts, double maxDistance, double maxHorizontal,
                                     double maxAngleFromDownwardDegrees) {
        return match(width,height,knobs,texts,maxDistance,maxHorizontal,maxAngleFromDownwardDegrees,SearchDirection.DOWN);
    }
    public static MatchReport match(int width, int height, List<Detection> knobs,
                                     List<Detection> texts, double maxDistance, double maxHorizontal,
                                     double maxAngleFromDownwardDegrees, SearchDirection direction) {
        return matchInternal(width,height,knobs,texts,maxDistance,maxHorizontal,maxAngleFromDownwardDegrees,direction,false);
    }
    /** App policy: one knob diagonal, and at least one third of the actual text polygon outside it. */
    public static MatchReport matchByKnobSize(int width,int height,List<Detection> knobs,List<Detection> texts,
                                              double angle,SearchDirection direction) {
        return matchInternal(width,height,knobs,texts,0,Double.NaN,angle,direction,true);
    }
    private static MatchReport matchInternal(int width,int height,List<Detection> knobs,List<Detection> texts,
                                             double maxDistance,double maxHorizontal,double maxAngleFromDownwardDegrees,
                                             SearchDirection direction,boolean knobRelative) {
        if (width <= 0 || height <= 0 || !Double.isFinite(maxDistance) || maxDistance < 0 ||
                (!Double.isNaN(maxHorizontal) && (!Double.isFinite(maxHorizontal) || maxHorizontal < 0)) ||
                !Double.isFinite(maxAngleFromDownwardDegrees) || maxAngleFromDownwardDegrees < 0 || maxAngleFromDownwardDegrees > 60 || direction == null)
            throw new IllegalArgumentException("匹配阈值或图片尺寸无效");
        validate(knobs, width, height); validate(texts, width, height);
        MatchReport report = new MatchReport(width, height, new ArrayList<>(knobs), new ArrayList<>(texts),
                                              maxDistance, maxHorizontal, maxAngleFromDownwardDegrees, direction,knobRelative);
        List<Detection> ordered = new ArrayList<>(knobs);
        ordered.sort(Comparator.comparingDouble((Detection item) -> -item.confidence).thenComparingInt(item -> item.id));
        List<Detection> orderedTexts = new ArrayList<>(texts);
        orderedTexts.sort(Comparator.comparingInt(item -> item.id));
        Map<Integer, Integer> owners = new HashMap<>();
        for (Detection knob : ordered) {
            double distanceLimit=report.maxDistanceFor(knob);
            MatchReport.Decision decision = new MatchReport.Decision(knob, report.decisions.size() + 1);
            MatchReport.Candidate best = null;
            boolean anyOutside = false, anyInDirection = false, anyGeometric = false;
            for (Detection text : orderedTexts) {
                double dx = text.cx - knob.cx, dy = text.cy - knob.cy;
                double distance = Math.hypot(dx, dy);
                List<String> reasons = new ArrayList<>();
                // Legacy absolute-distance callers retain their center rule; the app uses polygon outside area.
                boolean inside = text.cx >= knob.box[0] && text.cx <= knob.box[2]
                        && text.cy >= knob.box[1] && text.cy <= knob.box[3];
                double outsideRatio=TextOverlap.outsideRatio(knob,text);
                boolean outsideEnough=knobRelative?outsideRatio+1e-9>=MatchReport.MIN_TEXT_OUTSIDE_RATIO:!inside;
                if (!outsideEnough) reasons.add(knobRelative?"text_outside_area_insufficient":"text_center_inside_knob_box"); else anyOutside = true;
                boolean ahead = direction.ahead(dx,dy);
                if (!ahead) reasons.add(direction == SearchDirection.DOWN ? "not_below" : "not_in_search_direction");
                else if (outsideEnough) anyInDirection = true;
                if (ahead && direction.angle(dx,dy) > maxAngleFromDownwardDegrees + 1e-9)
                    reasons.add(direction == SearchDirection.DOWN ? "outside_downward_sector" : "outside_search_sector");
                if (distance > distanceLimit+1e-9*Math.max(1,distanceLimit)) reasons.add("distance_exceeded");
                if (!Double.isNaN(maxHorizontal) && direction.lateral(dx,dy) > maxHorizontal)
                    reasons.add(direction == SearchDirection.DOWN ? "horizontal_offset_exceeded" : "lateral_offset_exceeded");
                boolean geometric = reasons.isEmpty();
                anyGeometric |= geometric;
                Integer owner = owners.get(text.id);
                if (owner != null) reasons.add("text_already_used");
                MatchReport.Candidate candidate = new MatchReport.Candidate(text, dx, dy, distance, geometric, owner, reasons,distanceLimit,outsideRatio);
                decision.candidates.add(candidate);
                if (reasons.isEmpty() && (best == null || better(candidate, best))) best = candidate;
            }
            if (best != null) {
                best.selected = true;
                owners.put(best.text.id, knob.id);
                decision.pair = new MatchReport.Pair(knob, best.text, decision.rank, best.distance);
                report.matches.add(decision.pair);
            } else {
                decision.reason = texts.isEmpty() ? "no_texts" : !anyOutside ? (knobRelative?"no_text_with_required_outside_area":"no_text_outside_knob_box") : !anyInDirection ? (direction == SearchDirection.DOWN ? "no_text_below" : "no_text_in_search_direction") :
                        !anyGeometric ? "no_text_within_threshold" : "eligible_texts_already_used";
            }
            report.decisions.add(decision);
        }
        for (Detection text : orderedTexts) if (!owners.containsKey(text.id)) report.unmatchedTexts.add(text);
        return report;
    }
    private static boolean better(MatchReport.Candidate a, MatchReport.Candidate b) {
        int distanceOrder = Double.compare(a.distance, b.distance);
        if (distanceOrder != 0) return distanceOrder < 0;
        // Recognition happens after assignment: OCR confidence is not available here.
        return a.text.id < b.text.id;
    }
    private static void validate(List<Detection> items, int width, int height) {
        Set<Integer> ids = new HashSet<>();
        for (Detection item : items) {
            if (item.id < 0 || !ids.add(item.id) || !Double.isFinite(item.confidence) || item.confidence < 0 || item.confidence > 1 ||
                    item.box.length != 4 || !Double.isFinite(item.cx) || !Double.isFinite(item.cy))
                throw new IllegalArgumentException("检测数据无效");
            for (double value : item.box) if (!Double.isFinite(value)) throw new IllegalArgumentException("坐标无效");
            if (!(0 <= item.box[0] && item.box[0] <= item.cx && item.cx <= item.box[2] && item.box[2] <= width - 1 &&
                  0 <= item.box[1] && item.box[1] <= item.cy && item.cy <= item.box[3] && item.box[3] <= height - 1))
                throw new IllegalArgumentException("检测坐标超出图片");
        }
    }
    public static String describe(String reason) {
        if (reason == null) return "已匹配";
        switch (reason) {
            case "text_outside_area_insufficient": return "文本框位于当前旋钮框外的面积不足三分之一（含零面积文字框）";
            case "no_text_with_required_outside_area": return "没有框外面积达到三分之一的文本框";
            case "no_texts": return "没有检测到文字区域";
            case "text_center_inside_knob_box": return "文本框中心位于当前旋钮包围框内或边界上";
            case "no_text_outside_knob_box": return "没有中心位于当前旋钮包围框外的文本框";
            case "no_text_below": return "没有位于下方的文字区域";
            case "no_text_within_threshold": return "文字未同时满足重叠限制、所选方向、距离等条件";
            case "no_text_in_search_direction": return "所选方向没有文字区域";
            case "not_in_search_direction": return "文字中心没有位于所选搜索方向";
            case "outside_search_sector": return "文字偏离搜索轴，超出所选扇区";
            case "lateral_offset_exceeded": return "文字侧向偏移超出额外阈值";
            case "eligible_texts_already_used": return "合格文本已被高优先级旋钮占用";
            case "not_below": return "文字中心没有位于旋钮中心下方";
            case "outside_downward_sector": return "文字偏向侧面，超出向下搜索扇区";
            case "distance_exceeded": return "文字与旋钮距离超出阈值";
            case "horizontal_offset_exceeded": return "文字横向偏移超出额外阈值";
            case "text_already_used": return "文字已被其他旋钮占用";
            default: return "未被选中";
        }
    }
}
