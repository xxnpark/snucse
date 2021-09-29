import java.util.Scanner;

public class StudentIDValidator {
    static boolean isProperLength(String str) {return str.length() == 10;}

    static boolean hasProperDivision(String str) {return str.charAt(4) == '-';}

    static boolean hasProperDigits(String str) {
        for (int i = 0; i < str.length(); i++) {
            if (i != 4) {
                char ch = str.charAt(i);
                if (ch < '0' || ch > '9') {
                    return false;
                }
            }
        }
        return true;
    }

    static void validateStudentID(String input) {
        if (!isProperLength(input)) {
            System.out.println("The input length should be 10.");
        } else if (!hasProperDivision(input)) {
            System.out.println("Fifth character should be '-'.");
        } else if (!hasProperDigits(input)){
            System.out.println("Contains an invalid digit.");
        } else {
            System.out.println(input + " is a valid student ID.");
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        while (true) {
            String input = sc.next();
            if (input.equals("exit")) {
                sc.close();
                return;
            }
            validateStudentID(input);
        }
    }
}