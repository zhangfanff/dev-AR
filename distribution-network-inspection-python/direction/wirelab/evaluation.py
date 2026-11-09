import numpy as np
from scipy.optimize import linear_sum_assignment
from .io import axial_delta, directed_delta, references


def compare(
    item, annotation, result, method, mask_source, variant="default", tolerance=5.0
):
    rows = []
    for region in ("upper", "lower"):
        state = annotation["regions"][region]
        if state not in ("labeled", "no_wire"):
            continue
        truth = references(annotation, region) if state == "labeled" else []
        pred = result[region]["options"]
        pairs = []
        if truth and pred:
            cost = np.array(
                [
                    [abs(float(axial_delta(t["angle"], p["angle"]))) for p in pred]
                    for t in truth
                ]
            )
            ii, jj = linear_sum_assignment(cost)
            pairs = list(zip(ii.tolist(), jj.tolist()))
        matches = []
        for i, j in pairs:
            t, p = truth[i], pred[j]
            error = abs(float(axial_delta(t["angle"], p["angle"])))
            derror = (
                None
                if t["direction"] is None
                else abs(float(directed_delta(t["direction"], p["direction"])))
            )
            interval = p.get("interval")
            covered = False
            width = None
            if interval:
                delta = float(axial_delta(t["angle"], p["angle"]))
                covered = (
                    interval["low_offset_deg"] <= delta <= interval["high_offset_deg"]
                )
                width = interval["width_deg"]
            matches.append(
                {
                    "reference_group": t["group"],
                    "predicted_index": j,
                    "axis_error_deg": error,
                    "directed_error_deg": derror,
                    "within_tolerance": error <= tolerance,
                    "interval_available": interval is not None,
                    "interval_covered": covered,
                    "interval_width_deg": width,
                }
            )
        rows.append(
            {
                "key": item["key"],
                "role": item["role"],
                "source": item["source"],
                "region": region,
                "method": method,
                "mask_source": mask_source,
                "variant": variant,
                "tags": annotation.get("tags", []),
                "truth_count": len(truth),
                "predicted_count": len(pred),
                "matches": matches,
                "single_direction_region": len(truth) == 1,
                "primary_axis_error_deg": abs(
                    float(axial_delta(truth[0]["angle"], pred[0]["angle"]))
                )
                if len(truth) == 1 and pred
                else None,
                "directed_truth_count": sum(t["direction"] is not None for t in truth),
                "candidate_count": result[region]["candidate_count"],
                "rejected": result[region]["rejected"],
            }
        )
    return rows


def summarize(rows):
    matches = [m for r in rows for m in r["matches"]]
    errors = [m["axis_error_deg"] for m in matches]
    directed = [
        m["directed_error_deg"] for m in matches if m["directed_error_deg"] is not None
    ]
    ngt = sum(r["truth_count"] for r in rows)
    npred = sum(r["predicted_count"] for r in rows)
    tp = sum(m["within_tolerance"] for m in matches)
    intervals = [m for m in matches if m["interval_available"]]
    has_uncertainty = any(r["method"] == "probabilistic" for r in rows)
    positive = [r for r in rows if r["truth_count"] > 0]
    negative = [r for r in rows if r["truth_count"] == 0]
    single = [r for r in rows if r["single_direction_region"]]
    mean = lambda v: float(np.mean(v)) if v else None
    ratio = lambda a, b: float(a / b) if b else None
    return {
        "images": len({r["key"] for r in rows}),
        "regions": len(rows),
        "reference_directions": ngt,
        "predicted_directions": npred,
        "matched_directions": len(matches),
        "tolerance_deg": 5.0,
        "axis_MAE_deg_matched": mean(errors),
        "axis_median_deg_matched": float(np.median(errors)) if errors else None,
        "axis_P95_deg_matched": float(np.quantile(errors, 0.95)) if errors else None,
        "single_direction_top1_MAE_deg": mean(
            [
                r["primary_axis_error_deg"]
                for r in single
                if r["primary_axis_error_deg"] is not None
            ]
        ),
        "single_direction_top1_success_at_5deg": mean(
            [
                r["primary_axis_error_deg"] is not None
                and r["primary_axis_error_deg"] <= 5
                for r in single
            ]
        ),
        "mean_candidate_count_per_region": mean([r["candidate_count"] for r in rows]),
        "direction_precision_at_5deg": ratio(tp, npred),
        "direction_recall_at_5deg": ratio(tp, ngt),
        "positive_region_output_rate": mean(
            [bool(r["predicted_count"]) for r in positive]
        ),
        "negative_region_false_output_rate": mean(
            [bool(r["predicted_count"]) for r in negative]
        ),
        "directed_MAE_deg_matched": mean(directed),
        "opposite_direction_errors": sum(e > 90 for e in directed),
        "directed_reference_count": sum(r["directed_truth_count"] for r in rows),
        "CI_availability_all_references": ratio(len(intervals), ngt)
        if has_uncertainty
        else None,
        "CI_coverage_all_references": ratio(
            sum(m["interval_covered"] for m in matches), ngt
        )
        if has_uncertainty
        else None,
        "CI_coverage_available_matched": mean(
            [m["interval_covered"] for m in intervals]
        ),
        "CI_mean_width_deg": mean([m["interval_width_deg"] for m in intervals]),
        "scope": "Conditional matched angle error is accompanied by all-reference recall and interval availability; missing directions are not discarded.",
    }
