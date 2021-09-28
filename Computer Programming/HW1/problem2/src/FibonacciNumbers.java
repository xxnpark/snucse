public class FibonacciNumbers {
    public static void printFibonacciNumbers(int n) {
		// DO NOT change the skeleton code.
		// You can add codes anywhere you want.

        int sum = 0;
        int num1 = 0;
        int num2 = 0;

        for (int i = 0; i < n; i++) {
            int num;

            if (i == 0) num = 0;
            else if (i == 1) num = 1;
            else num = num1 + num2;

            System.out.print(num);
            if (i != n - 1) System.out.print(" ");
            sum += num;

            num1 = num2;
            num2 = num;
        }

        if (sum < 100000) System.out.print("\nsum = " + sum);
        else System.out.printf("\nlast five digits of sum = %05d", sum % 100000);
    }
}
