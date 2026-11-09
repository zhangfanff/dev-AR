package com.example.cabinetmatcher.camera.core;

/** One admitted FRAME at a time; its independent models may run in parallel.
 * Restarting invalidates all results from the old session. */
public final class FrameGate {
    public static final class Ticket {
        public final long generation, sequence, receivedNs;
        private Ticket(long generation, long sequence, long receivedNs) {
            this.generation = generation; this.sequence = sequence; this.receivedNs = receivedNs;
        }
    }
    private long generation, sequence;
    private boolean running;
    private Ticket active;
    public synchronized void start() { generation++; running = true; }
    public synchronized void stop() { generation++; running = false; }
    public synchronized Ticket tryBegin(long receivedNs) {
        if (!running || active != null) return null;
        active = new Ticket(generation, ++sequence, receivedNs);
        return active;
    }
    public synchronized boolean isCurrent(Ticket ticket) {
        return running && ticket != null && ticket.generation == generation;
    }
    public synchronized void finish(Ticket ticket) { if (active == ticket) active = null; }
    public synchronized boolean isRunning() { return running; }
    public synchronized boolean isBusy() { return active != null; }
}
