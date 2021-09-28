public class PrimeNumbers {
    public static void printPrimeNumbers(int m, int n) {
		// DO NOT change the skeleton code.
		// You can add codes anywhere you want.

        for (int k = m; k <= n; k++) {
            if (isPrimeNumber(k)) System.out.print(k + " ");
        }
    }

    public static boolean isPrimeNumber(int n) {
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (n % i == 0) return false;
        }
        return true;
    }
}
