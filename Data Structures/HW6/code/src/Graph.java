import java.util.*;

public class Graph { // Weighted Directed Graph
    HashMap<String, GraphNode> nodeIdSearch;
    HashMap<String, GraphNode> nodeNameSearch;
    ArrayList<GraphNode> nodes;
    static final GraphNode MAX_COST_NODE = new GraphNode();

    public Graph() {
        nodeIdSearch = new HashMap<>();
        nodeNameSearch = new HashMap<>();
        nodes = new ArrayList<>();
    }

    public void put(String id, String name, String line) {
        GraphNode newNode = new GraphNode(id, name, line);
        nodeIdSearch.put(id, newNode);
        if (nodeNameSearch.containsKey(name)) {
            GraphNode sameNameNode = nodeNameSearch.get(name);
            setDistance(sameNameNode.getId(), id, 5);
            setDistance(id, sameNameNode.getId(), 5);
            for (GraphNode node : sameNameNode.transferNodes) {
                setDistance(node.getId(), id, 5);
                setDistance(id, node.getId(), 5);
            }
            sameNameNode.setTransfer();
            sameNameNode.transferNodes.add(newNode);
        } else {
            nodeNameSearch.put(name, newNode);
        }
        nodes.add(newNode);
    }

    public void setDistance(String startId, String endId, long distance) {
        GraphNode startNode = nodeIdSearch.get(startId);
        GraphNode endNode = nodeIdSearch.get(endId);
        startNode.addAdjacent(endNode, distance);
    }

    public Pair<ArrayList<GraphNode>, Long> findShortestPath(String startName, String endName) { // Dijkstra Algorithm
        GraphNode startNode = nodeNameSearch.get(startName);
        GraphNode endNode = nodeNameSearch.get(endName);
        startNode.setCost(0L);
        HashMap<GraphNode, Long> startTransferNodeTimes = startNode.startNoTransfer();
        HashMap<GraphNode, Long> endTransferNodeTimes = endNode.endNoTransfer();

        for (GraphNode node : startNode.adjacentNodeTimes.keySet()) {
            node.setCost(startNode.adjacentNodeTimes.get(node));
        }

        ArrayList<GraphNode> selectedNodes = new ArrayList<>();
        selectedNodes.add(startNode);

        while (selectedNodes.size() != nodeIdSearch.size()) {
            GraphNode nextNode = MAX_COST_NODE;
            for (GraphNode node : nodes) {
                if (!selectedNodes.contains(node) && node.getCost() < nextNode.getCost()) {
                    nextNode = node;
                }
            }
            selectedNodes.add(nextNode);

            for (GraphNode node : nextNode.adjacentNodeTimes.keySet()) {
                if (!selectedNodes.contains(node) && nextNode.getCost() + nextNode.adjacentNodeTimes.get(node) < node.getCost()) {
                    node.setCost(nextNode.getCost() + nextNode.adjacentNodeTimes.get(node));
                    node.setPrevNode(nextNode);
                }
            }
        }

        long time = 0L;
        ArrayList<GraphNode> pathNodes = new ArrayList<>();
        pathNodes.add(endNode);
        GraphNode currNode = endNode;
        while (currNode.getPrevNode() != null) {
            time += currNode.getPrevNode().adjacentNodeTimes.get(currNode);
            currNode = currNode.getPrevNode();
            pathNodes.add(0, currNode);
        }
        time += startNode.adjacentNodeTimes.get(currNode);
        pathNodes.add(0, startNode);

        this.reset();
        startNode.startLetTransfer(startTransferNodeTimes);
        endNode.endLetTransfer(endTransferNodeTimes);

        return new Pair<>(pathNodes, time);
    }

    public void reset() {
        for (GraphNode node : nodes) {
            node.setCost(Long.MAX_VALUE);
            node.setPrevNode(null);
        }
    }
}
