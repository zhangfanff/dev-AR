"""Offline extension: weighted axial von Mises mixture + conditional block bootstrap.

The mixture replaces fixed-bin grouping, while extraction/quality filters remain
the same as the source baseline. BIC uses instance count, not pixel count.
Intervals are approximate conditional bootstrap intervals, not calibrated
guarantees or segmentation epistemic uncertainty. Coverage must be measured.
"""

import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e, i1e, logsumexp
from .io import axial_mean, axial_delta
from . import legacy


def concentration(r, cap=500.0):
    r = float(np.clip(r, 0, 1 - 1e-12))
    if r < 1e-8:
        return 0.0
    if r >= float(i1e(cap) / i0e(cap)):
        return cap
    return float(brentq(lambda k: float(i1e(k) / i0e(k)) - r, 0.0, cap))


def fit_mixture(angles, weights, k):
    z = np.deg2rad(np.asarray(angles) * 2)
    n = len(z)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum() * n
    if k == 1:
        means = np.array([np.deg2rad(axial_mean(angles, weights) * 2)])
    else:
        distance = np.abs(
            axial_delta(np.asarray(angles)[:, None], np.asarray(angles)[None, :])
        )
        i, j = np.unravel_index(np.argmax(distance), distance.shape)
        means = z[[i, j]].copy()
    kappas = np.full(k, 10.0)
    mix = np.full(k, 1 / k)
    old = -np.inf
    for _ in range(80):
        logp = (
            np.log(mix + 1e-15)[None, :]
            + kappas[None, :] * np.cos(z[:, None] - means[None, :])
            - np.log(2 * np.pi)
            - np.log(i0e(kappas))[None, :]
            - kappas[None, :]
        )
        ll = float(np.sum(weights * logsumexp(logp, axis=1)))
        resp = np.exp(logp - logsumexp(logp, axis=1)[:, None])
        wr = weights[:, None] * resp
        mass = wr.sum(axis=0)
        mix = mass / mass.sum()
        means = np.arctan2(
            (wr * np.sin(z[:, None])).sum(axis=0), (wr * np.cos(z[:, None])).sum(axis=0)
        )
        for j in range(k):
            r = abs(np.sum(wr[:, j] * np.exp(1j * z))) / max(mass[j], 1e-15)
            kappas[j] = concentration(r)
        if abs(ll - old) < 1e-7:
            break
        old = ll
    # Evaluate the final parameters, rather than the penultimate EM likelihood.
    logp = (
        np.log(mix + 1e-15)[None, :]
        + kappas[None, :] * np.cos(z[:, None] - means[None, :])
        - np.log(2 * np.pi)
        - np.log(i0e(kappas))[None, :]
        - kappas[None, :]
    )
    resp = np.exp(logp - logsumexp(logp, axis=1)[:, None])
    ll = float(np.sum(weights * logsumexp(logp, axis=1)))
    return {
        "means": np.rad2deg(means) / 2,
        "kappas": kappas,
        "mass": mix,
        "labels": resp.argmax(axis=1),
        "bic": float(-2 * ll + (3 * k - 1) * np.log(max(n, 2))),
    }


def selected_model(cs):
    a = [c["angle"] for c in cs]
    w = [c["weight"] for c in cs]
    models = [fit_mixture(a, w, 1)]
    if len(cs) >= 4:
        two = fit_mixture(a, w, 2)
        if (
            min(np.bincount(two["labels"], minlength=2)) >= 2
            and abs(float(axial_delta(*two["means"]))) >= 10
        ):
            models.append(two)
    return min(models, key=lambda x: x["bic"])


def grouped_angle(cs, cfg):
    return legacy.mean_in_bin(cs, True, cfg)


def point_blocks(c, blocks=8, per_block=256):
    points = c["_points"]
    proj = (points - points.mean(axis=0)) @ c["_axis"]
    span = float(np.ptp(proj))
    ids = np.minimum(
        blocks - 1, ((proj - proj.min()) / max(span, 1e-9) * blocks).astype(int)
    )
    result = []
    for i in range(blocks):
        p = points[ids == i]
        if not len(p):
            continue
        if len(p) > per_block:
            p = p[np.linspace(0, len(p) - 1, per_block, dtype=int)]
        result.append(p)
    return result


def interval_for(cs, angle, width, region, cfg, rng, reps):
    if reps <= 0:
        return {"interval": None, "bootstrap_valid_fraction": None}
    blocks = [point_blocks(c) for c in cs]
    samples = []
    for _ in range(reps):
        # Resample instances, then spatial blocks inside each selected instance.
        picked = rng.integers(0, len(cs), len(cs))
        draw = []
        for index in picked:
            b = blocks[index]
            if not b:
                continue
            points = np.concatenate(
                [b[j] for j in rng.integers(0, len(b), len(b))], axis=0
            )
            points = points[np.lexsort((points[:, 0], points[:, 1]))]
            c, _ = legacy.candidate(
                points, cs[index]["_confidence"], width, region, index, cfg
            )
            if c is not None:
                draw.append(c)
        if draw:
            samples.append(grouped_angle(draw, cfg))
    valid = len(samples) / reps
    if len(samples) < max(20, int(reps * 0.5)):
        return {
            "interval": None,
            "bootstrap_valid_fraction": valid,
            "interval_status": "insufficient_valid_resamples",
        }
    delta = axial_delta(np.asarray(samples), angle)
    lo, hi = np.quantile(delta, [0.025, 0.975])
    mean = axial_mean(samples)
    r = abs(np.mean(np.exp(1j * np.deg2rad(np.asarray(samples) * 2))))
    return {
        "interval": {
            "level": 0.95,
            "low_offset_deg": float(lo),
            "high_offset_deg": float(hi),
            "width_deg": float(hi - lo),
            "center_deg": angle,
            "kind": "conditional_spatial_block_and_instance_bootstrap_percentile",
        },
        "bootstrap_valid_fraction": valid,
        "bootstrap_replicates": reps,
        "sampling_von_mises": {
            "mean_axis_deg": mean,
            "kappa_on_doubled_angles": concentration(r, cap=1e6),
        },
        "interval_status": "available",
    }


def analyze(masks, width, height, cfg=legacy.Config(), bootstrap=200, seed=42):
    candidates, rejected = legacy.extract(masks, width, height, cfg, keep_points=True)
    rng = np.random.default_rng(seed)
    results = {}
    for region in ("upper", "lower"):
        cs = candidates[region]
        options = []
        if cs:
            fitted = selected_model(cs)
            for j in np.argsort(-fitted["mass"], kind="stable"):
                members = [c for i, c in enumerate(cs) if fitted["labels"][i] == j]
                if not members:
                    continue
                # Keep the platform's geometric sign convention for an output heading.
                angle = legacy.normalize_line(fitted["means"][j])
                if abs(angle) >= cfg.horizontal_correction:
                    corrected = grouped_angle(members, cfg)
                    angle = legacy.normalize_line(
                        abs(angle) if corrected >= 0 else -abs(angle)
                    )
                option = {
                    "angle": angle,
                    "direction": legacy.direction(angle, region),
                    "vote_count": len(members),
                    "weight": sum(c["weight"] for c in members),
                    "component_mass": float(fitted["mass"][j]),
                    "mixture_kappa": float(fitted["kappas"][j]),
                    "bic": fitted["bic"],
                    "source_indices": [c["source_index"] for c in members],
                }
                option.update(
                    interval_for(members, angle, width, region, cfg, rng, bootstrap)
                )
                options.append(option)
        results[region] = {
            "options": options,
            "candidate_count": len(cs),
            "rejected": rejected[region],
        }
    return results
