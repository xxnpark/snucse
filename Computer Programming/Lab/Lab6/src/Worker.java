import java.util.Queue;

public abstract class Worker {
    Queue<Work> workQueue;
    private static int numWorkers = 0;
    private final int id;
    private int sinceLastWork = 0;

    public abstract void run();

    Worker(Queue<Work> workQueue) {
        this.workQueue = workQueue;
        id = numWorkers++;
    }

    void report(String msg){
        System.out.printf("%s %s\n", this, msg);
    }

    @Override
    public String toString() {
        return "worker" + id;
    }

    public void incrementTime() {
        sinceLastWork++;
    }

    public void resetTime() {
        sinceLastWork = 0;
    }

    public int getTime() {
        return sinceLastWork;
    }
}

class Producer extends Worker {
    Producer(Queue<Work> workQueue) {
        super(workQueue);
    }

    @Override
    public void run() {
        if (workQueue.size() < 20){
            Work work = new Work();
            workQueue.add(work);
            this.report("produced " + work);
        } else {
            this.report("failed to produce work");
        }
    }
}

class Consumer extends Worker {
    Consumer(Queue<Work> workQueue) {
        super(workQueue);
    }

    @Override
    public void run() {
        if (Math.random() > 0.5) {
            Work work = workQueue.poll();
            if (work != null) {
                this.report("consumed " + work);
            } else {
                this.report("failed to consume work");
            }
        } else {
            this.report("failed to consume work");
        }
    }
}
