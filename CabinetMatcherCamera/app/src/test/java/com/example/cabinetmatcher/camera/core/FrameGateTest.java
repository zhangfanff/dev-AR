package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import static org.junit.Assert.*;

public class FrameGateTest {
    @Test public void pausedFramesAreNotAdmitted() {
        FrameGate gate = new FrameGate(); assertNull(gate.tryBegin(1)); gate.start();
        FrameGate.Ticket ticket = gate.tryBegin(2); assertNotNull(ticket); gate.stop();
        assertFalse(gate.isCurrent(ticket)); assertNull(gate.tryBegin(3)); gate.finish(ticket); assertFalse(gate.isBusy());
    }
    @Test public void busyFramesAreDroppedAndNeverQueued() {
        FrameGate gate = new FrameGate(); gate.start(); FrameGate.Ticket first = gate.tryBegin(100);
        for (int i = 0; i < 1000; i++) assertNull(gate.tryBegin(101 + i));
        gate.finish(first); FrameGate.Ticket newest = gate.tryBegin(5000);
        assertEquals(5000, newest.receivedNs); assertEquals(2, newest.sequence);
    }
    @Test public void stopAndRestartDiscardOldResultsButCannotOverlapInference() {
        FrameGate gate = new FrameGate(); gate.start(); FrameGate.Ticket old = gate.tryBegin(10);
        gate.stop(); gate.start(); assertFalse(gate.isCurrent(old)); assertNull(gate.tryBegin(20));
        gate.finish(old); FrameGate.Ticket fresh = gate.tryBegin(30); assertTrue(gate.isCurrent(fresh));
        assertTrue(fresh.generation > old.generation); gate.finish(old); assertTrue(gate.isBusy());
        gate.finish(fresh); assertFalse(gate.isBusy());
    }
    @Test public void concurrentCallersAdmitExactlyOneFrame() throws Exception {
        FrameGate gate = new FrameGate(); gate.start(); ExecutorService workers = Executors.newFixedThreadPool(8);
        CountDownLatch start = new CountDownLatch(1); List<Future<FrameGate.Ticket>> results = new ArrayList<>();
        try {
            for (int i = 0; i < 8; i++) results.add(workers.submit(() -> { start.await(); return gate.tryBegin(1); }));
            start.countDown(); int admitted = 0; FrameGate.Ticket accepted = null;
            for (Future<FrameGate.Ticket> result : results) { FrameGate.Ticket ticket = result.get(5, TimeUnit.SECONDS); if (ticket != null) { admitted++; accepted = ticket; } }
            assertEquals(1, admitted); gate.finish(accepted);
        } finally { workers.shutdownNow(); }
    }
}
