package com.example.cabinetmatcher.camera.core;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * Join native inference before releasing its input or closing models.
 * Never cancel these futures: Future.cancel(true) can mark a task done while JNI
 * continues reading the frame. Lifecycle cancellation is checked between stages.
 */
public final class InferenceTasks {
    private InferenceTasks() {}

    public static <T> T await(Future<T> future) throws Exception {
        boolean interrupted = false;
        try {
            T value;
            for (;;) {
                try {
                    value = future.get();
                    break;
                } catch (InterruptedException error) {
                    interrupted = true;
                } catch (ExecutionException error) {
                    Throwable cause = error.getCause();
                    if (cause instanceof Error) throw (Error) cause;
                    if (cause instanceof Exception) throw (Exception) cause;
                    throw new IllegalStateException(cause);
                }
            }
            if (interrupted) throw new InterruptedException("Interrupted while joining inference");
            return value;
        } finally { if (interrupted) Thread.currentThread().interrupt(); }
    }

    /** Used in finally, including when another branch has already failed. */
    public static void drain(Future<?> future) {
        if (future == null) return;
        boolean interrupted = false;
        try {
            for (;;) {
                try { future.get(); return; }
                catch (InterruptedException error) { interrupted = true; }
                catch (ExecutionException error) { return; }
            }
        } finally { if (interrupted) Thread.currentThread().interrupt(); }
    }
}
