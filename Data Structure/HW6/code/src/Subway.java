import java.io.*;
import java.util.*;

public class Subway {
    public static void main(String[] args) {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        try {
            Graph subway = FileToGraph(args[0]);
            while (true) {
                String input = br.readLine();
                if (input.compareTo("QUIT") == 0)
                    break;
                command(input, subway);
            }
        } catch (IOException e) {
            System.out.println("입력이 잘못되었습니다. 오류 : " + e.toString());
        }
    }

    private static void command(String input, Graph subway) {
        String[] inputList = input.split(" ");
        Pair<ArrayList<GraphNode>, Long> pathResult = subway.findShortestPath(inputList[0], inputList[1]);
        ArrayList<GraphNode> pathNodes = pathResult.key;
        long time = pathResult.value;

        StringBuilder sb = new StringBuilder();
        GraphNode prevNode = Graph.MAX_COST_NODE;
        boolean isTransfer = false;
        String beforeTransferId = null;

        for (GraphNode node : pathNodes) {
            if (prevNode.getName().equals(node.getName())) {
                isTransfer = true;
            } else if (prevNode != Graph.MAX_COST_NODE){
                if (isTransfer && beforeTransferId != null && !beforeTransferId.equals(prevNode.getId())) {
                    sb.append("[").append(prevNode.getName()).append("]").append(" ");
                    isTransfer = false;
                } else {
                    sb.append(prevNode.getName()).append(" ");
                }
                beforeTransferId = node.getId();
            }

            prevNode = node;
        }
        sb.append(inputList[1]);

        System.out.println(sb);
        System.out.println(time);
    }

    private static Graph FileToGraph(String filePath) throws IOException {
        Graph graph = new Graph();
        FileReader fr = new FileReader(filePath);
        BufferedReader br = new BufferedReader(fr);

        while (true) {
            String input = br.readLine();
            if (input.isEmpty()) {
                break;
            }
            String[] inputList = input.split(" ");
            graph.put(inputList[0], inputList[1], inputList[2]);

            /*
            for (GraphNode node : graph.nodes) {
                if (node.getName().equals(inputList[1])) {
                    graph.setDistance(node.getId(), inputList[0], 5);
                    graph.setDistance(inputList[0], node.getId(), 5);
                }
            }
            */
        }

        while (true) {
            String input = br.readLine();
            if (input == null) {
                break;
            }
            String[] inputList = input.split(" ");
            graph.setDistance(inputList[0], inputList[1], Integer.parseInt(inputList[2]));
        }

        return graph;
    }
}
