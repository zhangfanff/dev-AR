package com.example.cabinetmatcher.camera.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** Coordinate restoration and convex-polygon suppression; no native/Android dependencies. */
public final class TextRegionMerger {
    public static final double DUPLICATE_IOU = 0.5;
    private TextRegionMerger() {}

    public static final class Region {
        public final double[][] polygon;
        public final int rotationDegrees;
        public Region(double[][] polygon, int rotationDegrees) {
            this.polygon = polygon; this.rotationDegrees = rotationDegrees;
        }
    }

    /** Inverse of OpenCV clockwise rotations, using pixel coordinates (width/height minus one).
     * Keep the detected corner sequence: it also defines the recognition crop's orientation. */
    public static double[][] restore(double[][] polygon, int rotation, int width, int height) {
        double[][] result = new double[polygon.length][2];
        for (int i = 0; i < polygon.length; i++) {
            double x = polygon[i][0], y = polygon[i][1], originalX, originalY;
            switch (rotation) {
                case 0: originalX = x; originalY = y; break;
                case 90: originalX = y; originalY = height - 1 - x; break;
                case 180: originalX = width - 1 - x; originalY = height - 1 - y; break;
                case 270: originalX = width - 1 - y; originalY = x; break;
                default: throw new IllegalArgumentException("Unsupported rotation: " + rotation);
            }
            result[i][0] = Math.max(0, Math.min(width - 1, originalX));
            result[i][1] = Math.max(0, Math.min(height - 1, originalY));
        }
        return result;
    }

    /** Prefer larger complete boxes; equal-area ties preserve pass order (original first).
     * Suppress only across passes, without changing the detector's own same-pass segmentation. */
    public static List<double[][]> merge(List<Region> candidates) {
        List<Region> ordered = new ArrayList<>(candidates);
        ordered.sort(Comparator.comparingDouble((Region r) -> -area(r.polygon)));
        List<Region> kept = new ArrayList<>();
        for (Region candidate : ordered) {
            if (area(candidate.polygon) <= 1e-6) continue;
            boolean duplicate = false;
            for (Region existing : kept) {
                if (candidate.rotationDegrees != existing.rotationDegrees
                        && iou(candidate.polygon, existing.polygon) >= DUPLICATE_IOU) {
                    duplicate = true; break;
                }
            }
            if (!duplicate) kept.add(candidate);
        }
        // Stable reading-position IDs after merging, independent of the winning pass's first corner.
        kept.sort(Comparator.comparingDouble((Region r) -> min(r.polygon, 1))
                .thenComparingDouble(r -> min(r.polygon, 0)));
        List<double[][]> result = new ArrayList<>();
        for (Region region : kept) result.add(region.polygon);
        return result;
    }

    private static double min(double[][] polygon, int axis) {
        double value = Double.POSITIVE_INFINITY;
        for (double[] point : polygon) value = Math.min(value, point[axis]);
        return value;
    }
    private static double max(double[][] polygon, int axis) {
        double value = Double.NEGATIVE_INFINITY;
        for (double[] point : polygon) value = Math.max(value, point[axis]);
        return value;
    }
    public static double iou(double[][] a, double[][] b) {
        if (max(a, 0) <= min(b, 0) || max(b, 0) <= min(a, 0)
                || max(a, 1) <= min(b, 1) || max(b, 1) <= min(a, 1)) return 0;
        double areaA = area(a), areaB = area(b);
        if (areaA <= 1e-6 || areaB <= 1e-6) return 0;
        List<double[]> intersection = new ArrayList<>(Arrays.asList(a));
        double winding = signedArea(b) >= 0 ? 1 : -1;
        for (int edge = 0; edge < b.length && !intersection.isEmpty(); edge++) {
            double[] start = b[edge], end = b[(edge + 1) % b.length];
            List<double[]> output = new ArrayList<>();
            double[] previous = intersection.get(intersection.size() - 1);
            double previousSide = winding * cross(start, end, previous);
            for (double[] current : intersection) {
                double currentSide = winding * cross(start, end, current);
                if ((currentSide >= 0) != (previousSide >= 0)) {
                    double fraction = previousSide / (previousSide - currentSide);
                    output.add(new double[]{previous[0] + fraction * (current[0] - previous[0]),
                            previous[1] + fraction * (current[1] - previous[1])});
                }
                if (currentSide >= 0) output.add(current);
                previous = current; previousSide = currentSide;
            }
            intersection = output;
        }
        double overlap = area(intersection.toArray(new double[0][]));
        return Math.max(0, Math.min(1, overlap / (areaA + areaB - overlap)));
    }
    private static double cross(double[] start, double[] end, double[] point) {
        return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]);
    }
    private static double signedArea(double[][] polygon) {
        double twice = 0;
        for (int i = 1; i + 1 < polygon.length; i++) twice += cross(polygon[0], polygon[i], polygon[i + 1]);
        return twice * 0.5;
    }
    private static double area(double[][] polygon) { return Math.abs(signedArea(polygon)); }
}
