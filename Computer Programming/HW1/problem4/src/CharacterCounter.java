public class CharacterCounter {
    public static void countCharacter(String str) {
		// DO NOT change the skeleton code.
		// You can add codes anywhere you want.

        for (char c = 65; c < 91; c++) {
            int upperCount = 0;
            int lowerCount = 0;
            for (int i = 0; i < str.length(); i++) {
                if (str.charAt(i) == c) upperCount++;
                if (str.charAt(i) == c + 32) lowerCount++;
            }
            if (upperCount > 0) {
                if (lowerCount > 0) {
                    System.out.printf("%c: %d times, ", c, upperCount);
                    printCount((char)(c + 32), lowerCount);
                }
                else printCount(c, upperCount);
            } else if (lowerCount > 0) {
                printCount((char)(c + 32), lowerCount);
            }
        }
    }

    private static void printCount(char character, int count) {
        System.out.printf("%c: %d times\n", character, count);
    }
}
