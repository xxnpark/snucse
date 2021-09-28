public class DecreasingString {
    public static void printLongestDecreasingSubstringLength(String inputString) {
		// DO NOT change the skeleton code.
		// You can add codes anywhere you want.

        char prev = inputString.charAt(0);

        int len = 1;
        int maxLen = 1;

        for (int i = 1; i < inputString.length(); i++) {
            if (prev > inputString.charAt(i)) len++;
            else {
                if (len > maxLen) maxLen = len;
                len = 1;
            }
            prev = inputString.charAt(i);
        }

        if (len > maxLen) maxLen = len;

        System.out.println(maxLen);
    }
}
