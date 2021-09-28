public class NumberCounter {
    public static void countNumbers(String str0, String str1, String str2) {
		// DO NOT change the skeleton code.
		// You can add codes anywhere you want.
        String mul = Integer.toString(Integer.parseInt(str0) * Integer.parseInt(str1) * Integer.parseInt(str2));
        for (char c = 48; c < 58; c++) {
            int count = 0;
            for (int i = 0; i < mul.length(); i++) {
                if (mul.charAt(i) == c) count++;
            }
            if (count > 0) printNumberCount(c - '0', count);
        }
        System.out.println("length: " + mul.length());

    }

    private static void printNumberCount(int number, int count) {
        System.out.printf("%d: %d times\n", number, count);
    }
}
