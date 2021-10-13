import java.util.List;

public class Scheduler<T extends Worker>{
    private static final int waitms = 400;
    private List<T> workers;

    public Scheduler(List<T> workers) {
        this.workers = workers;
    }

    T schedule() {
        return workers.get(0);
    }

    T schedule(int index) {
        if (index >= 0 && index < workers.size()) {
            return workers.get(index);
        } else {
            return schedule();
        }
    }

    T scheduleRandom() {
        return workers.get((int)(Math.random() * workers.size()));
    }

    T scheduleFair() {
        T oldestWorker = null;
        int oldestTime = -1;
        for (T worker: workers) {
            if (worker.getTime() > oldestTime) {
                oldestTime = worker.getTime();
                oldestWorker = worker;
            }
        }
        incrementTime();
        oldestWorker.resetTime();
        return oldestWorker;
    }

    private void incrementTime() {
        for (Worker worker: workers) {
            worker.incrementTime();
        }
    }

    static void delay() {
        try {
            Thread.sleep(waitms);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
