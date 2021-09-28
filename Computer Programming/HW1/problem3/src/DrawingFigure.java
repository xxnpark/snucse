public class DrawingFigure {
    public static void drawFigure(int n) {
		// DO NOT change the skeleton code.
		// You can add codes anywhere you want.

        for (int i = 0; i <= n; i++) {
            System.out.print(" ".repeat(2 * (n - i)));
            System.out.print("* ".repeat(2 * i));
            System.out.print("*");
            System.out.println(" ".repeat(2 * (n - i)));
        }

        for (int i = n-1; i >=0; i--) {
            System.out.print(" ".repeat(2 * (n - i)));
            System.out.print("* ".repeat(2 * i));
            System.out.print("*");
            System.out.println(" ".repeat(2 * (n - i)));
        }
    }
}
