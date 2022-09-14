import java.util.*;

public class GraphNode {
    private String id;
    private String name;
    private String line;
    private long cost;
    boolean isTransfer = false;
    ArrayList<GraphNode> transferNodes;
    HashMap<GraphNode, Long> adjacentNodeTimes;
    private GraphNode prevNode;

    public GraphNode() {
        this("", "", "");
    }

    public GraphNode(String id, String name, String line) {
        this.id = id;
        this.name = name;
        this.line = line;
        this.cost = Long.MAX_VALUE;
        this.transferNodes = new ArrayList<>();
        this.adjacentNodeTimes = new HashMap<>();
        this.prevNode = null;
    }

    public void addAdjacent(GraphNode node, long time) {
        adjacentNodeTimes.put(node, time);
    }

    public HashMap<GraphNode, Long> startNoTransfer() {
        HashMap<GraphNode, Long> transferNodeTimes = new HashMap<>();
        for (GraphNode node : transferNodes) {
            transferNodeTimes.put(node, adjacentNodeTimes.get(node));
            adjacentNodeTimes.put(node, 0L);
        }
        return transferNodeTimes;
    }

    public HashMap<GraphNode, Long> endNoTransfer() {
        HashMap<GraphNode, Long> transferNodeTimes = new HashMap<>();
        for (GraphNode node : transferNodes) {
            transferNodeTimes.put(node, node.adjacentNodeTimes.get(this));
            node.adjacentNodeTimes.put(this, 0L);
        }
        return transferNodeTimes;
    }

    public void startLetTransfer(HashMap<GraphNode, Long> transferNodeTimes) {
        for (GraphNode node : transferNodeTimes.keySet()) {
            adjacentNodeTimes.put(node, transferNodeTimes.get(node));
        }
    }

    public void endLetTransfer(HashMap<GraphNode, Long> transferNodeTimes) {
        for (GraphNode node : transferNodeTimes.keySet()) {
            node.adjacentNodeTimes.put(this, transferNodeTimes.get(node));
        }
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String getLine() {
        return line;
    }

    public long getCost() {
        return cost;
    }

    public void setCost(Long cost) {
        this.cost = cost;
    }

    public GraphNode getPrevNode() {
        return prevNode;
    }

    public void setPrevNode(GraphNode prevNode) {
        this.prevNode = prevNode;
    }

    public boolean isTransfer() {
        return isTransfer;
    }

    public void setTransfer() {
        isTransfer = true;
    }
}
