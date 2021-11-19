import cpta.Grader;
import cpta.environment.Compiler;
import cpta.environment.Executer;
import cpta.exam.ExamSpec;
import cpta.exam.Problem;
import cpta.exam.Student;
import cpta.exam.TestCase;


import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class ExtraTestCases2 {
    public static void main(String[] args) {
        if(!(new File("data")).exists()){
            System.err.println("Data path doesn't exist! Make sure that the working directory is set to HW4/problem1");
            return;
        }
        System.out.println("Sub-problem 1 Original Test Cases");
        testSubProblem1Original();

        System.out.println("\n\nSub-problem 2 Original Test Cases");
        testSubProblem2Original();

        System.out.println("\n\nSub-problem 2 Extra Test Cases (by Hwang Yae Eun):");
        testSubProblem2Hwang();

        System.out.println("\n\nSub-problem 2 Extra Test Cases (by Ahn Jae Woo):");
        testSubProblem2Ahn();

    }

    static void testSubProblem1Original() {
        String submissionDirPath = "data/exam-simple/";
        resetData(submissionDirPath);

        Student s1 = new Student("2020-12345", "Youngki Lee");
        Student s2 = new Student("2020-12346", "Kiroong Choe");
        List<Student> students = new ArrayList<>(); students.add(s1); students.add(s2);

        TestCase t1 = new TestCase("1", "1.in", "1.out", 50);
        TestCase t2 = new TestCase("2", "2.in", "2.out", 50);
        List<TestCase> testCases = new ArrayList<>();
        testCases.add(t1);
        testCases.add(t2);

        Problem p1 = new Problem(
                "problem1", "data/test-cases/test-exam-simple/problem1/",
                testCases, "Problem1.sugo", null, null
        );
        Problem p2 = new Problem(
                "problem2", "data/test-cases/test-exam-simple/problem2/",
                testCases, "Problem2.sugo", null, null
        );
        List<Problem> problems = new ArrayList<>(); problems.add(p1); problems.add(p2);

        ExamSpec examSpec = new ExamSpec(problems, students);
        Grader grader = new Grader(new Compiler(), new Executer());
        Map<String,Map<String,List<Double>>> result = grader.gradeSimple(examSpec, submissionDirPath);

        Map<String,Map<String,List<Double>>> expectedResult = new HashMap<>();
        expectedResult = Map.of("2020-12345", Map.of("problem1", Arrays.asList(50.0, 50.0),
                        "problem2", Arrays.asList(50.0, 50.0)),
                "2020-12346", Map.of("problem1", Arrays.asList(0.0, 0.0),
                        "problem2", Arrays.asList(50.0, 0.0)));

        testAndPrintResult(result, expectedResult);
    }

    static void testSubProblem2Original() {
        String submissionDirPath = "data/exam-robust/";
        resetData(submissionDirPath);


        Student s1 = new Student("2020-12345", "Youngki Lee");
        Student s2 = new Student("2020-12346", "Kiroong Choe");
        Student s3 = new Student("2020-12347", "Clumsy Kang");
        Student s4 = new Student("2020-12348", "Multi-file Jung");
        Student s5 = new Student("2020-12349", "Troller Park");
        Student s6 = new Student("2020-12350", "Dropper Kim");
        List<Student> students = new ArrayList<>();
        students.add(s1); students.add(s2); students.add(s3);
        students.add(s4); students.add(s5); students.add(s6);

        TestCase t2_1 = new TestCase("1", "1.in", "1.out", 50);
        TestCase t2_2 = new TestCase("2", "2.in", "2.out", 50);
        List<TestCase> testCases2 = new ArrayList<>(); testCases2.add(t2_1); testCases2.add(t2_2);
        TestCase t3_1 = new TestCase("1", "1.in", "1.out", 50);
        TestCase t3_2 = new TestCase("2", "2.in", "2.out", 50);
        TestCase t3_3 = new TestCase("3", "3.in", "3.out", 50);
        List<TestCase> testCases3 = new ArrayList<>();
        testCases3.add(t3_1); testCases3.add(t3_2); testCases3.add(t3_3);

        Set<String> judgingTypesA = new TreeSet<>(); judgingTypesA.add(Problem.LEADING_WHITESPACES);
        Set<String> judgingTypesB = new TreeSet<>(); judgingTypesB.add(Problem.IGNORE_WHITESPACES);
        Set<String> judgingTypesC = new TreeSet<>(); judgingTypesC.add(Problem.CASE_INSENSITIVE);

        Problem p1 = new Problem(
                "problem1", "data/test-cases/test-exam-robust/problem1/",
                testCases2, "Problem1.sugo", null, judgingTypesA
        );
        Problem p2 = new Problem(
                "problem2", "data/test-cases/test-exam-robust/problem2/",
                testCases2, "Problem2.sugo", null, judgingTypesB
        );
        Problem p3 = new Problem(
                "problem3", "data/test-cases/test-exam-robust/problem3/",
                testCases3, "Wrapper.sugo",
                "data/test-cases/test-exam-robust/problem3/wrappers/",judgingTypesC
        );
        List<Problem> problems = new ArrayList<>(); problems.add(p1); problems.add(p2); problems.add(p3);

        ExamSpec examSpec = new ExamSpec(problems, students);

        Grader grader = new Grader(new Compiler(), new Executer());
        Map<String,Map<String,List<Double>>> result = grader.gradeRobust(examSpec, submissionDirPath);

        Map<String,Map<String,List<Double>>> expectedResult = new HashMap<>();
        expectedResult = Map.of("2020-12345", Map.of("problem1", Arrays.asList(50.0, 50.0),
                        "problem2", Arrays.asList(50.0, 50.0),
                        "problem3", Arrays.asList(50.0, 50.0, 50.0)),
                "2020-12346", Map.of("problem1", Arrays.asList(0.0, 0.0),
                        "problem2", Arrays.asList(50.0, 0.0),
                        "problem3", Arrays.asList(0.0, 0.0, 0.0)),
                "2020-12347", Map.of("problem1", Arrays.asList(50.0, 50.0),
                        "problem2", Arrays.asList(50.0, 50.0),
                        "problem3", Arrays.asList(50.0, 50.0, 50.0)),
                "2020-12348", Map.of("problem1", Arrays.asList(50.0, 50.0),
                        "problem2", Arrays.asList(50.0, 50.0),
                        "problem3", Arrays.asList(50.0, 50.0, 50.0)),
                "2020-12349", Map.of("problem1", Arrays.asList(50.0, 50.0),
                        "problem2", Arrays.asList(25.0, 25.0),
                        "problem3", Arrays.asList(0.0, 0.0, 0.0)),
                "2020-12350", Map.of("problem1", Arrays.asList(0.0, 0.0),
                        "problem2", Arrays.asList(0.0, 0.0),
                        "problem3", Arrays.asList(0.0, 0.0, 0.0)));

        testAndPrintResult(result, expectedResult);
    }
    static void testSubProblem2Hwang() {
        String submissionDirPath = "extra-data-2/extra-exam-robust/";
        resetData(submissionDirPath);

        Student s1 = new Student("2020-12345", "A");
        Student s2 = new Student("2020-12346", "B");
        Student s3 = new Student("2020-12347", "C");
        Student s4 = new Student("2020-12348", "D");
        Student s5 = new Student("2020-12349", "E");
        List<Student> students = new ArrayList<>();
        students.add(s1); students.add(s2); students.add(s3);
        students.add(s4); students.add(s5);

        TestCase t2_1 = new TestCase("1", "1.in", "1.out", 50);
        List<TestCase> testCases2 = new ArrayList<>(); testCases2.add(t2_1);

        Set<String> judgingTypesA = new TreeSet<>();
        judgingTypesA.add(Problem.LEADING_WHITESPACES); judgingTypesA.add(Problem.CASE_INSENSITIVE); judgingTypesA.add(Problem.IGNORE_SPECIAL_CHARACTERS);
        Set<String> judgingTypesB = new TreeSet<>();
        judgingTypesB.add(Problem.IGNORE_SPECIAL_CHARACTERS);
        Set<String> judgingTypesC = null;
        Set<String> judgingTypesD = new TreeSet<>();

        Problem p1 = new Problem(
                "problem1", "extra-data-2/extra-test-cases/problem1/",
                testCases2, "Problem1.sugo", null, judgingTypesA
        );
        Problem p2 = new Problem(
                "problem2", "extra-data-2/extra-test-cases/problem2/",
                testCases2, "Problem2.sugo", null, judgingTypesB
        );
        Problem p3 = new Problem(
                "problem3", "extra-data-2/extra-test-cases/problem3/",
                testCases2, "Problem3.sugo",
                null, judgingTypesC
        );
        Problem p4 = new Problem(
                "problem4", "extra-data-2/extra-test-cases/problem4/",
                testCases2, "Problem4.sugo",
                null, judgingTypesD
        );
        List<Problem> problems = new ArrayList<>(); problems.add(p1); problems.add(p2); problems.add(p3); problems.add(p4);

        ExamSpec examSpec = new ExamSpec(problems, students);

        Grader grader = new Grader(new Compiler(), new Executer());
        Map<String,Map<String,List<Double>>> result = grader.gradeRobust(examSpec, submissionDirPath);

        Map<String,Map<String,List<Double>>> expectedResult = new HashMap<>();
        expectedResult = Map.of("2020-12345", Map.of("problem1", Arrays.asList(50.0),
                        "problem2", Arrays.asList(50.0),
                        "problem3", Arrays.asList(50.0),
                        "problem4", Arrays.asList(50.0)),
                "2020-12346", Map.of("problem1", Arrays.asList(50.0),
                        "problem2", Arrays.asList(50.0),
                        "problem3", Arrays.asList(0.0),
                        "problem4", Arrays.asList(0.0)),
                "2020-12347", Map.of("problem1", Arrays.asList(50.0),
                        "problem2", Arrays.asList(50.0),
                        "problem3", Arrays.asList(0.0),
                        "problem4", Arrays.asList(0.0)),
                "2020-12348", Map.of("problem1", Arrays.asList(0.0),
                        "problem2", Arrays.asList(0.0),
                        "problem3", Arrays.asList(0.0),
                        "problem4", Arrays.asList(0.0)),
                "2020-12349", Map.of("problem1", Arrays.asList(0.0),
                        "problem2", Arrays.asList(0.0),
                        "problem3", Arrays.asList(0.0),
                        "problem4", Arrays.asList(50.0)));

        testAndPrintResult(result, expectedResult);

        System.out.println("problem 1 : problem.judgingTypes includes multiple cases (Problem.LEADING_WHITESPACES, Problem.CASE_INSENSITIVE, Problem.IGNORE_SPECIAL_CHARACTERS)");
        System.out.println("problem 2 : problem.judgingTypes includes only Problem.IGNORE_SPECIAL_CHARACTERS");
        System.out.println("problem 3 : problem.judgingTypes is null");
        System.out.println("problem 4 : problem.judgingTypes is empty");
        System.out.println("If 2020-12347's score for problem 2 is 0.0, files inside the additional directory did not overwrite the file with same name outside the additional directory");
        System.out.println("2020-12348 has student submission directory and problem submission directory for problem 3, but is missing file");
        System.out.println("If 2020-12349's score for problem 4 is 0.0, unnecessary yo file was not overwritten by result of compiling sugo file of same name");

    }

    static void testSubProblem2Ahn(){
        String submissionDirPath = "extra-data-2/extra-exam-robust/";
        resetData(submissionDirPath);

        Student s1 = new Student("2021-12345", "Mr. Mistake");
        List<Student> students = new ArrayList<>();
        students.add(s1);
        System.out.println("2021-12345 problem1: LEADING_WHITESPACES shouldn't ignore trailing whitespaces");
        System.out.println("2021-12345 problem2: You should use TestCase.score, not the constant '50'. If you got full score (40), it means you didn't cut the score by half.");
        System.out.println("2021-12345 problem3: You should cut the score by half if there is at least one yo file without sugo file EVEN IF IT IS NOT RELATED TO TARGET FILE.");
        System.out.println("2021-12345 problem4: You should check the existence of sugo file AFTER moving the files out from the additional directory");
        System.out.println("2021-12345 problem5: IGNORE_WHITESPACES should ignore leading, MIDDLE, and trailing whitespaces");
        System.out.println("2021-12345 problem6: A list of scores for individual test cases should be in ascending order of test case ID.");

        Set<String> judgingTypesLeadingWhitespaces = new TreeSet<>();
        judgingTypesLeadingWhitespaces.add(Problem.LEADING_WHITESPACES);
        Set<String> judgingTypesIgnoreWhitespaces = new TreeSet<>();
        judgingTypesIgnoreWhitespaces.add(Problem.IGNORE_WHITESPACES);

        TestCase t1 = new TestCase("1", "1.in", "1.out", 50);
        List<TestCase> testCases1 = new ArrayList<>();
        testCases1.add(t1);
        Problem p1 = new Problem("problem1", "extra-data-2/extra-test-cases/problem1/", testCases1, "Problem1.sugo", null, judgingTypesLeadingWhitespaces);

        TestCase t2 = new TestCase("1", "1.in", "1.out", 40);
        List<TestCase> testCases2 = new ArrayList<>();
        testCases2.add(t2);
        Problem p2 = new Problem("problem2", "extra-data-2/extra-test-cases/problem2/", testCases2, "Problem2.sugo", null, null);

        TestCase t3 = new TestCase("1", "1.in", "1.out", 50);
        List<TestCase> testCases3 = new ArrayList<>();
        testCases3.add(t3);
        Problem p3 = new Problem("problem3", "extra-data-2/extra-test-cases/problem3/", testCases3, "Problem3.sugo", null, null);

        TestCase t4 = new TestCase("1", "1.in", "1.out", 50);
        List<TestCase> testCases4 = new ArrayList<>();
        testCases4.add(t4);
        Problem p4 = new Problem("problem4", "extra-data-2/extra-test-cases/problem4/", testCases4, "Problem4.sugo", null, null);

        TestCase t5 = new TestCase("1", "1.in", "1.out", 50);
        List<TestCase> testCases5 = new ArrayList<>();
        testCases5.add(t5);
        Problem p5 = new Problem("problem5", "extra-data-2/extra-test-cases/problem5/", testCases5, "Problem5.sugo", null, judgingTypesIgnoreWhitespaces);
        List<Problem> problems = new ArrayList<>();

        TestCase t6_1 = new TestCase("2", "1.in", "1.out", 50);
        TestCase t6_2 = new TestCase("1", "2.in", "2.out", 75);
        List<TestCase> testCases6 = new ArrayList<>();
        testCases6.add(t6_1);
        testCases6.add(t6_2);
        Problem p6 = new Problem("problem6", "extra-data-2/extra-test-cases/problem6/", testCases6, "Problem6.sugo", null, null);

        problems.add(p1);
        problems.add(p2);
        problems.add(p3);
        problems.add(p4);
        problems.add(p5);
        problems.add(p6);
        ExamSpec examSpec = new ExamSpec(problems, students);

        Grader grader = new Grader(new Compiler(), new Executer());

        Map<String,Map<String,List<Double>>> result = grader.gradeRobust(examSpec, submissionDirPath);

        Map<String,Map<String,List<Double>>> expectedResult = new HashMap<>();
        expectedResult = Map.of("2021-12345", Map.of("problem1", Arrays.asList(0.0),
                "problem2", Arrays.asList(20.0),
                "problem3", Arrays.asList(25.0),
                "problem4", Arrays.asList(25.0),
                "problem5", Arrays.asList(50.0),
                "problem6", Arrays.asList(75.0, 50.0)));

        testAndPrintResult(result, expectedResult);
    }

    static void testAndPrintResult(Map<String,Map<String,List<Double>>> result, Map<String,Map<String,List<Double>>> expected) {
        boolean testResult = test(result, expected);
        printOX(testResult);

        if (!testResult) {
            System.out.println("Your Result : ");
            printResult(result);
            System.out.println("Expected Result : ");
            printResult(expected);
        }
    }

    static void printResult(Map<String,Map<String,List<Double>>> result) {
        if (result == null) {
            return;
        }

        SortedSet<String> studentIdSet = new TreeSet<String>(result.keySet());
        for(String studentId : studentIdSet) {
            System.out.println(studentId);
            Map<String,List<Double>> studentResult = result.get(studentId);
            SortedSet<String> problemIdSet = new TreeSet<String>(studentResult.keySet());
            for(String problemId : problemIdSet) {
                List<Double> problemResult = studentResult.get(problemId);
                System.out.print("\t" + problemId + ": ");
                for(double score : problemResult) {
                    System.out.print(score + " ");
                }
                System.out.println();
            }
        }
    }

    static void printOX(boolean result) {
        System.out.println(result ? "O" : "X");
    }

    static boolean test(Map<String,Map<String,List<Double>>> result, Map<String,Map<String,List<Double>>> expected) {
        if (result == null) {
            return false;
        }

        if (!(expected.keySet()).equals(result.keySet())) {
            return false;
        }

        for (String id : result.keySet()) {
            Map<String, List<Double>> gradesResult = result.get(id);
            Map<String, List<Double>> gradesExpected = expected.get(id);
            if (!gradesResult.keySet().equals(gradesExpected.keySet())) {
                return false;
            }
            for (String problem : gradesResult.keySet()) {
                List<Double> res = gradesResult.get(problem);
                List<Double> exp = gradesExpected.get(problem);
                if (res.size() != exp.size()) {
                    return false;
                }
                for (int i=0; i<res.size(); i++) {
                    if (!res.get(i).equals(exp.get(i))) {
                        System.out.printf("student %s %s %d-th different %f %f\n", id, problem, i, res.get(i), exp.get(i));
                        return false;
                    }
                }
            }
        }

        return true;
    }

    static void resetData(String origPath) {
        // Delete directory
        deleteDirectory(new File(origPath));


        // Copy new data from backup
        String backupPath = null;
        if (origPath.contains("extra-exam-robust")) backupPath = origPath.replace("extra-exam-robust", "extra-exam-robust-backup/");
        if (origPath.contains("exam-robust")) backupPath = origPath.replace("exam-robust", "exam-robust-backup/");
        if (origPath.contains("exam-simple")) backupPath = origPath.replace("exam-simple", "exam-simple-backup/");

        try {
            Path sourceDir = Paths.get(backupPath);
            Path destinationDir = Paths.get(origPath);
            // Traverse the file tree and copy each file/directory.
            Files.walk(sourceDir)
                    .forEach(sourcePath -> {
                        try {
                            Path targetPath = destinationDir.resolve(sourceDir.relativize(sourcePath));
                            Files.copy(sourcePath, targetPath);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static void deleteDirectory(File file) {
        if (!file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] entries = file.listFiles();
            if (entries != null) {
                for (File entry : entries) {
                    deleteDirectory(entry);
                }
            }
        }
        file.delete();
    }


}
