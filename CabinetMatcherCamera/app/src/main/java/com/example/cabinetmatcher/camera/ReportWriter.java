package com.example.cabinetmatcher.camera;

import com.example.cabinetmatcher.camera.core.Detection;
import com.example.cabinetmatcher.camera.core.GreedyMatcher;
import com.example.cabinetmatcher.camera.core.MatchReport;
import com.example.cabinetmatcher.camera.core.TimingReport;
import com.example.cabinetmatcher.camera.core.TextOverlap;
import org.json.JSONArray;
import org.json.JSONObject;

final class ReportWriter {
    static String json(MatchReport report) throws Exception {
        double sx = (double) report.sourceWidth / report.width, sy = (double) report.sourceHeight / report.height;
        JSONObject output = new JSONObject();
        output.put("image_name", report.imageName);
        output.put("image_size", size(report.sourceWidth, report.sourceHeight));
        output.put("processing_image_size", size(report.width, report.height));
        output.put("coordinate_system", "EXIF-oriented original image pixels; top-left origin; x right; y down");
        output.put("strategy", "match_before_recognition");
        output.put("algorithm", "knob-confidence-first one-to-one geometry matching, then OCR of assigned regions only");
        output.put("execution_order", new JSONArray(new String[]{"whole_image_text_detection", "knob_detection", "geometry_matching", "selected_region_recognition"}));
        output.put("matching_uses_recognition_confidence", false);
        output.put("retry_other_region_on_recognition_failure", false);
        output.put("elapsed_ms", report.elapsedMs);
        output.put("timings", timing(report.timing));
        output.put("parameters", new JSONObject().put("distance_basis",report.knobRelative?"per_knob_box_diagonal":"absolute_processing_pixels")
                .put("text_detection_mode", report.timing.textDetectionMode.name())
                .put("text_detection_size_mode", report.timing.textDetectionSizeMode.name())
                .put("max_distance_knob_diagonal_multiplier",report.knobRelative?1:JSONObject.NULL)
                .put("min_text_outside_knob_area_ratio",report.knobRelative?MatchReport.MIN_TEXT_OUTSIDE_RATIO:JSONObject.NULL)
                .put("overlap_area_geometry","actual text polygon intersected with current axis-aligned YOLO box")
                .put("max_distance_ratio_of_diagonal", report.knobRelative?JSONObject.NULL:report.maxDistance / Math.hypot(report.width, report.height))
                .put("max_distance_px", report.knobRelative?JSONObject.NULL:report.maxDistance * Math.hypot(report.sourceWidth, report.sourceHeight) / Math.hypot(report.width, report.height))
                .put("processing_max_distance_px", report.knobRelative?JSONObject.NULL:report.maxDistance)
                .put("text_center_must_be_outside_knob_box", !report.knobRelative)
                .put("knob_box_boundary_counts_as_inside", !report.knobRelative)
                .put("text_center_outside_rule", report.knobRelative?JSONObject.NULL:"text.cx < knob.x1 OR text.cx > knob.x2 OR text.cy < knob.y1 OR text.cy > knob.y2; applied separately to each current knob")
                .put("text_area_outside_rule",report.knobRelative?"(text_polygon_area - intersection_with_current_knob_box_area) / text_polygon_area >= 1/3; zero-area text rejected":JSONObject.NULL)
                .put("max_lateral_processing_px", Double.isNaN(report.maxHorizontal) ? JSONObject.NULL : report.maxHorizontal)
                .put("max_horizontal_px", Double.isNaN(report.maxHorizontal) || report.searchDirection == com.example.cabinetmatcher.camera.core.SearchDirection.LEFT || report.searchDirection == com.example.cabinetmatcher.camera.core.SearchDirection.RIGHT || report.searchDirection == com.example.cabinetmatcher.camera.core.SearchDirection.ALL ? JSONObject.NULL : report.maxHorizontal * sx)
                .put("search_direction", report.searchDirection.name())
                .put("sector_half_angle_degrees", report.maxAngleFromDownwardDegrees)
                .put("max_angle_from_downward_degrees", report.searchDirection == com.example.cabinetmatcher.camera.core.SearchDirection.DOWN ? report.maxAngleFromDownwardDegrees : JSONObject.NULL)
                .put("sector_rule", "positive projection on selected axis and atan2(abs(perpendicular projection), forward projection) <= half angle; ALL uses union of four axes; closest available text wins once")
                .put("below_rule", report.searchDirection == com.example.cabinetmatcher.camera.core.SearchDirection.DOWN ? "dy > 0 and degrees(atan2(abs(dx), dy)) <= sector_half_angle_degrees; processing image coordinates" : JSONObject.NULL));
        output.put("summary", new JSONObject().put("knob_count", report.knobs.size()).put("text_region_count", report.texts.size())
                .put("geometric_match_count", report.matches.size()).put("recognized_match_count", report.recognizedMatches())
                .put("recognition_failed_match_count", report.matches.size() - report.recognizedMatches())
                .put("recognition_attempt_count", report.timing.textRegions.size()).put("recognition_model_calls", report.timing.recognitionModelCalls)
                .put("skipped_region_count", report.unmatchedTexts.size()).put("unmatched_knob_count", report.knobs.size() - report.matches.size())
                .put("unmatched_text_region_count", report.unmatchedTexts.size()));
        JSONArray pairs = new JSONArray(), decisions = new JSONArray(), unmatchedKnobs = new JSONArray(), unmatchedTexts = new JSONArray(), order = new JSONArray();
        for (MatchReport.Pair pair : report.matches) pairs.put(pair(pair, sx, sy,report)
                .put("search_axis", report.searchDirection.axis(pair.text.cx-pair.knob.cx,pair.text.cy-pair.knob.cy).name()));
        for (MatchReport.Decision decision : report.decisions) {
            order.put(decision.knob.id);
            JSONObject entry = new JSONObject().put("processing_rank", decision.rank).put("knob", detection(decision.knob, sx, sy))
                    .put("status", decision.pair == null ? "unmatched" : "geometrically_matched")
                    .put("max_distance_processing_px",report.maxDistanceFor(decision.knob))
                    .put("max_distance_px",sourceLimit(report,decision.knob,sx,sy))
                    .put("matched_region_id", decision.pair == null ? JSONObject.NULL : decision.pair.text.id)
                    .put("recognition_status", decision.pair == null ? JSONObject.NULL : decision.pair.text.recognitionStatus)
                    .put("reason", decision.reason == null ? JSONObject.NULL : decision.reason);
            JSONArray candidates = new JSONArray();
            for (MatchReport.Candidate candidate : decision.candidates) candidates.put(new JSONObject()
                    .put("region_id", candidate.text.id).put("text", candidate.text.text == null ? JSONObject.NULL : candidate.text.text)
                    .put("dx_px", round(candidate.dx * sx)).put("dy_px", round(candidate.dy * sy))
                    .put("distance_px", round(Math.hypot(candidate.dx * sx, candidate.dy * sy)))
                    .put("processing_distance_px", round(candidate.distance))
                    .put("max_distance_processing_px",report.maxDistanceFor(decision.knob))
                    .put("text_outside_knob_area_ratio",TextOverlap.outsideRatio(decision.knob,candidate.text))
                    .put("geometrically_eligible", candidate.geometric)
                    .put("processing_angle_from_downward_degrees", round(Math.toDegrees(Math.atan2(Math.abs(candidate.dx), candidate.dy))))
                    .put("search_axis", report.searchDirection.axis(candidate.dx,candidate.dy).name())
                    .put("processing_angle_from_search_axis_degrees", round(report.searchDirection.angle(candidate.dx,candidate.dy)))
                    .put("processing_lateral_offset_px", round(report.searchDirection.lateral(candidate.dx,candidate.dy)))
                    .put("available", candidate.reasons.isEmpty()).put("selected", candidate.selected)
                    .put("occupied_by_knob_id", candidate.owner == null ? JSONObject.NULL : candidate.owner)
                    .put("rejection_reasons", new JSONArray(candidate.reasons))
                    .put("rejection_reason_descriptions", reasonDescriptions(candidate.reasons)));
            entry.put("candidate_evaluations", candidates); decisions.put(entry);
            if (decision.pair == null) unmatchedKnobs.put(new JSONObject().put("processing_rank", decision.rank)
                    .put("knob", detection(decision.knob, sx, sy)).put("reason", decision.reason)
                    .put("reason_description", GreedyMatcher.describe(decision.reason)));
        }
        for (Detection text : report.unmatchedTexts) unmatchedTexts.put(new JSONObject().put("text_region", detection(text, sx, sy))
                .put("reason", "not_assigned_recognition_skipped").put("reason_description", "未被几何匹配选中，跳过文字识别"));
        output.put("matches", pairs).put("knob_results", decisions).put("processing_order", order)
                .put("unmatched_knobs", unmatchedKnobs).put("unmatched_text_regions", unmatchedTexts);
        JSONArray knobs = new JSONArray(), texts = new JSONArray();
        for (Detection knob : report.knobs) knobs.put(detection(knob, sx, sy));
        for (Detection text : report.texts) texts.put(detection(text, sx, sy));
        output.put("input_results", new JSONObject().put("knobs", new JSONObject().put("count", report.knobs.size()).put("detections", knobs))
                .put("ocr", new JSONObject().put("count", report.texts.size()).put("text_regions", texts)));
        return output.toString(2);
    }
    private static JSONObject size(int w, int h) throws Exception { return new JSONObject().put("width", w).put("height", h); }
    private static JSONArray reasonDescriptions(java.util.List<String> reasons) {
        JSONArray output = new JSONArray();
        for (String reason : reasons) output.put(GreedyMatcher.describe(reason));
        return output;
    }
    private static JSONObject timing(TimingReport timing) throws Exception {
        JSONObject stages = milliseconds(timing);
        JSONArray passes = new JSONArray();
        for (TimingReport.DetectionPass pass : timing.detectionPasses) passes.put(new JSONObject()
                .put("clockwise_rotation_degrees", pass.rotationDegrees).put("region_count", pass.regionCount)
                .put("source_image_size", pass.timing.detectionSourceWidth > 0
                        ? size(pass.timing.detectionSourceWidth,pass.timing.detectionSourceHeight) : JSONObject.NULL)
                .put("detector_input_size", pass.timing.detectionInputWidth > 0
                        ? size(pass.timing.detectionInputWidth,pass.timing.detectionInputHeight) : JSONObject.NULL)
                .put("stages_ms", milliseconds(pass.timing)));
        JSONArray regions = new JSONArray();
        for (TimingReport.TextCost cost : timing.textRegions) regions.put(new JSONObject()
                .put("region_index", cost.regionIndex).put("text_id", cost.textId == null ? JSONObject.NULL : cost.textId)
                .put("text", cost.text).put("confidence", "error".equals(cost.recognitionStatus) ? JSONObject.NULL : cost.confidence).put("accepted", cost.accepted)
                .put("recognition_status", cost.recognitionStatus).put("error", cost.error == null ? JSONObject.NULL : cost.error)
                .put("recognition_model_calls", cost.timing.recognitionModelCalls)
                .put("crop_width", cost.cropWidth).put("crop_height", cost.cropHeight)
                .put("stages_ms", milliseconds(cost.timing)));
        return new JSONObject().put("unit", "ms").put("clock", "System.nanoTime monotonic wall clock")
                .put("text_detection_mode", timing.textDetectionMode.name())
                .put("text_detection_size_mode", timing.textDetectionSizeMode.name())
                .put("raw_text_region_count", timing.rawTextRegionCount)
                .put("merged_text_region_count", timing.detectedTextRegionCount)
                .put("text_detection_passes", passes)
                .put("cross_direction_merge", timing.textDetectionMode == com.example.cabinetmatcher.camera.core.TextDetectionMode.FOUR_DIRECTIONS
                        ? "original-frame convex polygon IoU >= 0.5; larger area first; same-pass boxes preserved" : "disabled")
                .put("models_reused", timing.modelsReused).put("angle_classifier_enabled", timing.angleClassifierEnabled)
                .put("detected_text_region_count", timing.detectedTextRegionCount)
                .put("recognition_attempt_count", timing.textRegions.size()).put("recognition_model_calls", timing.recognitionModelCalls)
                .put("skipped_region_count", timing.detectedTextRegionCount - timing.textRegions.size())
                .put("scope", "image_load was measured during selection; analysis.total is button click to UI content update; pipeline.total preserves elapsed_ms scope; excludes gallery interaction, export and GPU frame presentation")
                .put("totals_note", "Parent totals include child stages. Do not sum all keys; other includes cleanup, scheduling and instrumentation overhead. Optional or reused stages are zero.")
                .put("stages_ms", stages).put("text_regions", regions);
    }
    private static JSONObject milliseconds(TimingReport timing) throws Exception {
        JSONObject stages = new JSONObject();
        for (String key : timing.stages.keySet()) stages.put(key, Math.round(timing.millis(key) * 1000.0) / 1000.0);
        return stages;
    }
    private static JSONObject detection(Detection item, double sx, double sy) throws Exception {
        JSONObject data = new JSONObject().put("id", item.id)
                .put("confidence", item.textRegion && (!item.recognitionAttempted || "error".equals(item.recognitionStatus)) ? JSONObject.NULL : item.confidence)
                .put("bbox_xyxy", new JSONArray(new double[]{round(item.box[0] * sx), round(item.box[1] * sy), round(item.box[2] * sx), round(item.box[3] * sy)}))
                .put("center_xy", new JSONArray(new double[]{round(item.cx * sx), round(item.cy * sy)}));
        if (item.textRegion) data.put("type", "text_region").put("text", item.text == null ? JSONObject.NULL : item.text)
                .put("recognition_attempted", item.recognitionAttempted).put("recognition_status", item.recognitionStatus)
                .put("recognition_error", item.recognitionError == null ? JSONObject.NULL : item.recognitionError);
        else data.put("class_id", 0).put("class_name", "knob");
        if (item.polygon != null) {
            JSONArray polygon = new JSONArray();
            for (double[] point : item.polygon) polygon.put(new JSONArray(new double[]{round(point[0] * sx), round(point[1] * sy)}));
            data.put("polygon", polygon);
        }
        return data;
    }
    private static JSONObject pair(MatchReport.Pair pair, double sx, double sy,MatchReport report) throws Exception {
        double dx = (pair.text.cx - pair.knob.cx) * sx, dy = (pair.text.cy - pair.knob.cy) * sy;
        return new JSONObject().put("processing_rank", pair.rank).put("knob", detection(pair.knob, sx, sy))
                .put("text_region", detection(pair.text, sx, sy)).put("recognition_status", pair.text.recognitionStatus)
                .put("max_distance_processing_px",report.maxDistanceFor(pair.knob))
                .put("max_distance_px",sourceLimit(report,pair.knob,sx,sy))
                .put("text_outside_knob_area_ratio",TextOverlap.outsideRatio(pair.knob,pair.text))
                .put("dx_px", round(dx)).put("dy_px", round(dy))
                .put("distance_px", round(Math.hypot(dx, dy)));
    }
    private static double sourceLimit(MatchReport report,Detection knob,double sx,double sy) {
        return report.knobRelative?Math.hypot((knob.box[2]-knob.box[0])*sx,(knob.box[3]-knob.box[1])*sy)
                :report.maxDistance*Math.hypot(report.sourceWidth,report.sourceHeight)/Math.hypot(report.width,report.height);
    }
    private static double round(double value) { return Math.round(value * 10000) / 10000.0; }
}
