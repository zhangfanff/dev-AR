"""Regional PCA direction estimation with fixed-bin voting.

Float32 conversions and endpoint selection follow WireOffsetCalculatorDPCA."""

from dataclasses import dataclass, asdict
import math
import numpy as np


def f(x):
    return float(np.float32(x))


@dataclass(frozen=True)
class Config:
    min_area: int = 50
    min_aspect: float = 2.0
    min_residual_aspect: float = 1.2
    bin_size: float = 15.0
    bin_offset: float = 0.0
    second_min_votes: int = 2
    alternative_min_diff: float = 10.0
    horizontal_wrap: float = 75.0
    horizontal_correction: float = 75.0
    sample_ratio: float = 0.2
    side_confidence: float = 0.25
    median_dx_ratio: float = 0.02
    median_dx_pixels: float = 6.0
    upper_end: float = 0.6
    lower_start: float = 0.4
    min_pca_samples: int = 10
    extend: float = 1.2
    median_correction: bool = True
    side_fallback: bool = True
    merge_horizontal: bool = True


def normalize_line(angle):
    a = f(angle)
    while a > 90:
        a = f(a - 180)
    while a <= -90:
        a = f(a + 180)
    return f(max(f(-89.999), min(f(89.999), a)))


def direction(angle, region):
    a = f(angle + 180) if region == "lower" else f(angle)
    while a > 180:
        a = f(a - 360)
    while a <= -180:
        a = f(a + 360)
    return a


def line_distance(a, b):
    d = abs(f(a - b))
    return min(d, 180 - d)


def stats_of(points, width, cfg):
    x = points[:, 0]
    y = points[:, 1]
    cx = f(float(x.mean()))
    ratio = f(cx / width)
    ratio = max(0.0, min(1.0, ratio))
    bias = f(ratio - 0.5)
    side = f(min(1.0, f(abs(bias) * 2)))
    if y.max() - y.min() <= 2:
        top = bottom = cx
    else:
        order = np.argsort(y, kind="stable")
        count = max(1, int(f(len(points) * f(cfg.sample_ratio))))
        top = f(np.median(x[order[:count]]))
        bottom = f(np.median(x[order[-count:]]))
    return {
        "center_x_ratio": ratio,
        "side_bias": bias,
        "side_confidence": side,
        "top_median_x": top,
        "bottom_median_x": bottom,
        "median_dx": f(top - bottom),
    }


def fit(points, cfg):
    n = len(points)
    if n < cfg.min_pca_samples:
        return None
    mean = points.mean(axis=0)
    q = points - mean
    xx = float(np.sum(q[:, 0] * q[:, 0]) / (n - 1))
    xy = float(np.sum(q[:, 0] * q[:, 1]) / (n - 1))
    yy = float(np.sum(q[:, 1] * q[:, 1]) / (n - 1))
    trace = xx + yy
    det = xx * yy - xy * xy
    disc = max(0.0, trace * trace - 4 * det)
    l1 = (trace + math.sqrt(disc)) / 2
    l2 = (trace - math.sqrt(disc)) / 2
    axis = (
        np.array([l1 - yy, xy])
        if abs(xy) > 1e-10
        else np.array([1.0, 0.0])
        if xx > yy
        else np.array([0.0, 1.0])
    )
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return None
    axis = axis / norm
    proj = q @ axis
    lo = int(np.argmin(proj))
    hi = int(np.argmax(proj))
    if not np.isfinite(proj[[lo, hi]]).all() or proj[hi] <= proj[lo]:
        return None
    p1 = mean + (points[lo] - mean) * cfg.extend
    p2 = mean + (points[hi] - mean) * cfg.extend
    top, bottom = (p1, p2) if p1[1] < p2[1] else (p2, p1)
    v = top - bottom
    length = float(np.linalg.norm(v))
    raw = (
        0.0
        if length == 0
        else f(math.degrees(math.acos(abs(max(-1.0, min(1.0, -v[1] / length))))))
    )
    raw = raw if v[0] < 0 else -raw
    raw = f(max(f(-89.999), min(f(89.999), raw)))
    residual = float((proj[hi] - proj[lo]) / max(2 * math.sqrt(max(l2, 0.0)), 1.0))
    linearity = max(0.0, min(1.0, (l1 - max(l2, 0.0)) / l1)) if l1 > 1e-6 else 0.0
    return {
        "raw": raw,
        "length": length,
        "residual_aspect": residual,
        "linearity": linearity,
        "axis": axis,
    }


def candidate(
    points, confidence, width, region, index, cfg=Config(), keep_points=False
):
    if len(points) < cfg.min_area:
        return None, "area"
    p = fit(points, cfg)
    if p is None:
        return None, "pca"
    aspect = p["length"] / max(2 * float(points[:, 0].std(ddof=0)), 1.0)
    if aspect < cfg.min_aspect:
        return None, "aspect"
    if p["residual_aspect"] < cfg.min_residual_aspect:
        return None, "residual_aspect"
    stats = stats_of(points, width, cfg)
    raw = p["raw"]
    angle_abs = abs(raw)
    threshold = max(f(cfg.median_dx_pixels), f(width * f(cfg.median_dx_ratio)))
    sign = 1 if raw >= 0 else -1
    source = "ENDPOINT"
    if cfg.median_correction and abs(stats["median_dx"]) >= threshold:
        sign = 1 if stats["median_dx"] < 0 else -1
        source = "MEDIAN_X"
    if (
        cfg.side_fallback
        and angle_abs >= f(cfg.horizontal_correction)
        and stats["side_confidence"] >= f(cfg.side_confidence)
    ):
        side_sign = 1 if stats["side_bias"] < 0 else -1
        if sign != side_sign or abs(stats["median_dx"]) < threshold:
            sign = side_sign
            source = "SIDE_BIAS_NEAR_HORIZONTAL"
    confidence = float(confidence)
    if not np.isfinite(confidence):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))
    weight = confidence * p["linearity"]
    if not np.isfinite(weight) or weight <= 0:
        return None, "weight"
    c = {
        "angle": normalize_line(angle_abs if sign >= 0 else -angle_abs),
        "raw_angle": raw,
        "weight": weight,
        "stats": stats,
        "sign_source": source,
        "region": region,
        "source_index": index,
        "area": len(points),
        "aspect": aspect,
        "residual_aspect": p["residual_aspect"],
        "linearity": p["linearity"],
    }
    if keep_points:
        c["_points"] = points
        c["_confidence"] = confidence
        c["_axis"] = p["axis"]
    return c, None


def extract(masks, width, height, cfg=Config(), keep_points=False, mode="both"):
    candidates = {"upper": [], "lower": []}
    rejected = {"upper": {}, "lower": {}}
    # Kotlin multiplies Int by Float here, not Double.
    upper = int(f(height * f(cfg.upper_end)))
    lower = int(f(height * f(cfg.lower_start)))
    for index, (mask, confidence) in enumerate(masks):
        if mask.shape != (height, width):
            raise ValueError("Mask size does not match image.")
        for region, start, end in [("upper", 0, upper), ("lower", lower, height)]:
            if mode != "both" and mode != region:
                continue
            y, x = np.nonzero(mask[start:end])
            y = y + start
            if len(x) == 0:
                continue
            points = np.column_stack([x, y]).astype(float)
            c, why = candidate(
                points, confidence, width, region, index, cfg, keep_points
            )
            if c is None:
                rejected[region][why] = rejected[region].get(why, 0) + 1
            else:
                candidates[region].append(c)
    return candidates, rejected


def mean_in_bin(cs, horizontal, cfg):
    total = sum(c["weight"] for c in cs)
    plain = f(sum(c["angle"] * c["weight"] for c in cs) / total)
    if not horizontal:
        return plain
    ss = sum(math.sin(math.radians(c["angle"] * 2)) * c["weight"] for c in cs)
    cc = sum(math.cos(math.radians(c["angle"] * 2)) * c["weight"] for c in cs)
    angle = (
        plain
        if ss == 0 and cc == 0
        else normalize_line(f(math.degrees(math.atan2(ss, cc)) / 2))
    )
    bias = f(sum(c["stats"]["side_bias"] * c["weight"] for c in cs) / total)
    confidence = f(min(1.0, f(abs(bias) * 2)))
    if (
        cfg.side_fallback
        and abs(angle) >= f(cfg.horizontal_correction)
        and confidence >= f(cfg.side_confidence)
    ):
        angle = normalize_line(abs(angle) if bias < 0 else -abs(angle))
    return angle


def vote(cs, region, cfg=Config()):
    if not cs:
        return []
    n = int(180 / cfg.bin_size) + 1
    horizontal = n

    def bin_of(c):
        if cfg.merge_horizontal and abs(c["angle"]) >= f(cfg.horizontal_wrap):
            return horizontal
        value = c["angle"] + 90
        if cfg.bin_offset:
            value = (value + cfg.bin_offset) % 180
        return max(0, min(n - 1, int(value / cfg.bin_size)))

    bins = [[] for _ in range(n + 1)]
    for c in cs:
        bins[bin_of(c)].append(c)
    ranking = sorted(range(len(bins)), key=lambda i: (-len(bins[i]), i))

    def option(index):
        angle = mean_in_bin(bins[index], index == horizontal, cfg)
        return {
            "angle": angle,
            "direction": direction(angle, region),
            "vote_count": len(bins[index]),
            "weight": sum(c["weight"] for c in bins[index]),
            "bin_index": index,
            "source_indices": [c["source_index"] for c in bins[index]],
        }

    options = [option(ranking[0])]
    if len(bins[ranking[1]]) >= cfg.second_min_votes:
        other = option(ranking[1])
        if (
            line_distance(other["angle"], options[0]["angle"])
            >= cfg.alternative_min_diff
        ):
            options.append(other)
    return options


def analyze(masks, width, height, cfg=Config(), mode="both"):
    cs, rejected = extract(masks, width, height, cfg, mode=mode)
    return {
        r: {
            "options": vote(cs[r], r, cfg),
            "candidate_count": len(cs[r]),
            "rejected": rejected[r],
        }
        for r in ("upper", "lower")
    }
