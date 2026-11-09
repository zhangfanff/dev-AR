package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.Assert.*;

public class InferenceTasksTest {
    @Test public void interruptedJoinStillWaitsForNativeWorkAndRestoresInterrupt() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        CountDownLatch started = new CountDownLatch(1), release = new CountDownLatch(1), exited = new CountDownLatch(1);
        AtomicBoolean restored = new AtomicBoolean(), nativeFinished = new AtomicBoolean();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Future<Integer> nativeTask = worker.submit(() -> { started.countDown(); release.await(); nativeFinished.set(true); return 42; });
        Thread coordinator = new Thread(() -> {
            try { InferenceTasks.await(nativeTask); failure.set(new AssertionError("Expected interrupt")); }
            catch (InterruptedException expected) { restored.set(Thread.currentThread().isInterrupted()); }
            catch (Throwable error) { failure.set(error); }
            finally { exited.countDown(); }
        });
        try {
            assertTrue(started.await(2, TimeUnit.SECONDS)); coordinator.start(); coordinator.interrupt();
            assertFalse("Must not free native input on interrupt", exited.await(100, TimeUnit.MILLISECONDS));
            release.countDown(); assertTrue(exited.await(2, TimeUnit.SECONDS));
            assertTrue(nativeFinished.get()); assertTrue(restored.get()); assertNull(failure.get());
        } finally { release.countDown(); coordinator.join(2000); worker.shutdown(); }
    }

    @Test public void branchFailureDrainsSiblingBeforeFrameRelease() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        CountDownLatch started = new CountDownLatch(1), release = new CountDownLatch(1), exited = new CountDownLatch(1);
        AtomicBoolean siblingFinished = new AtomicBoolean(), frameFreed = new AtomicBoolean();
        Future<?> sibling = worker.submit(() -> { started.countDown(); release.await(); siblingFinished.set(true); return null; });
        Thread coordinator = new Thread(() -> {
            try { throw new IllegalStateException("OCR failed"); }
            catch (IllegalStateException expected) { /* MainActivity/CabinetPipeline cleanup pattern. */ }
            finally { InferenceTasks.drain(sibling); frameFreed.set(siblingFinished.get()); exited.countDown(); }
        });
        try {
            assertTrue(started.await(2, TimeUnit.SECONDS)); coordinator.start();
            assertFalse(exited.await(100, TimeUnit.MILLISECONDS)); assertFalse(frameFreed.get());
            release.countDown(); assertTrue(exited.await(2, TimeUnit.SECONDS)); assertTrue(frameFreed.get());
        } finally { release.countDown(); coordinator.join(2000); worker.shutdown(); }
    }

    @Test public void cleanupPreservesExistingInterruptAndWaitsForFailingNativeTask() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        CountDownLatch release = new CountDownLatch(1), exited = new CountDownLatch(1);
        AtomicBoolean restored = new AtomicBoolean();
        Future<?> nativeTask = worker.submit(() -> { release.await(); throw new IllegalArgumentException("Native error"); });
        Thread coordinator = new Thread(() -> {
            Thread.currentThread().interrupt(); InferenceTasks.drain(nativeTask);
            restored.set(Thread.currentThread().isInterrupted()); exited.countDown();
        });
        try {
            coordinator.start(); assertFalse(exited.await(100, TimeUnit.MILLISECONDS));
            release.countDown(); assertTrue(exited.await(2, TimeUnit.SECONDS)); assertTrue(restored.get());
        } finally { release.countDown(); coordinator.join(2000); worker.shutdown(); }
    }

    @Test public void workerErrorIsReportedAfterWorkerHasExited() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        AtomicBoolean exited = new AtomicBoolean();
        OutOfMemoryError expected = new OutOfMemoryError("simulated");
        try {
            Future<?> task = worker.submit(() -> { try { throw expected; } finally { exited.set(true); } });
            try { InferenceTasks.await(task); fail("Expected error"); }
            catch (OutOfMemoryError actual) { assertSame(expected, actual); assertTrue(exited.get()); }
            InferenceTasks.drain(task);
        } finally { worker.shutdown(); }
    }
}
