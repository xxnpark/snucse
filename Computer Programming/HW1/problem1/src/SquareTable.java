public class SquareTable {
    public static void printSquareTable(int n) {
        // DO NOT change the skeleton code.
        // You can add codes anywhere you want.

        int k = 1;
        while (k*k <= n) {
            printOneSquare(k, k*k);
            k++;
        }
    }

    private static void printOneSquare(int a, int b) {
        System.out.printf("%d times %d = %d\n", a, a, b);
    }
}
