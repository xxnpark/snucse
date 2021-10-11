import java.util.*;
import java.io.*;

public class Main {
    static ArrayList<Student> students = new ArrayList<>();
    static boolean IS_TEST_MODE = false;

    public static void main(String[] args) {
        if (args.length >= 1 && args[0].equals("--test")) IS_TEST_MODE = true;

        Scanner scanner = null;
        try {
            if (IS_TEST_MODE) {
                File file = new File("test-script.txt");
                scanner = new Scanner(file);
            } else {
                scanner = new Scanner(System.in);
            }
        } catch (FileNotFoundException e) {
            System.exit(1);
        }

        while (true) {
            String input = scanner.nextLine();
            try {
                Request request = new Request(input);
                execute(request);
            } catch (AppException e) {
                System.out.println(e);
            }
        }
    }

    //< @brief 파싱된 request 객체를 받아서 실제 동작을 실행하는 함수입니다.
    //< @param request 실행할 요청 객체
    static void execute(Request request) throws AppException {
        // TODO implement
    }
}

