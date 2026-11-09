package com.example.cabinetmatcher.camera;

/** MediaPipe VIDEO requires strictly increasing millisecond timestamps. */
final class FrameClock {
    private long last = -1;
    long next(long uptimeMs) { last = Math.max(last + 1, uptimeMs); return last; }
}
