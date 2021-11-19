package cpta;

import cpta.environment.Compiler;
import cpta.environment.Executer;
import cpta.exam.ExamSpec;
import cpta.exam.Problem;
import cpta.exam.Student;
import cpta.exam.TestCase;
import cpta.exceptions.CompileErrorException;
import cpta.exceptions.FileSystemRelatedException;
import cpta.exceptions.InvalidFileTypeException;
import cpta.exceptions.RunTimeErrorException;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Grader {
    Compiler compiler;
    Executer executer;

    public Grader(Compiler compiler, Executer executer) {
        this.compiler = compiler;
        this.executer = executer;
    }

    public Map<String,Map<String, List<Double>>> gradeSimple(ExamSpec examSpec, String submissionDirPath) {
        List<Problem> problems = examSpec.problems;
        List<Student> students = examSpec.students;

        Map<String, Map<String, List<Double>>> grades = new HashMap<>();

        for (Student student : students) {
            Map<String, List<Double>> studentGrades = new HashMap<>();

            for (Problem problem : problems) {
                List<Double> problemStudentGrades = new LinkedList<>();

                try {
                    List<TestCase> testCases = problem.testCases;

                    String basePath = submissionDirPath + student.id + "/" + problem.id + "/";
                    String filePath = basePath + problem.targetFileName;

                    compiler.compile(filePath);

                    for (TestCase testCase : testCases) {
                        double score;

                        String targetFilePath = filePath.split("\\.")[0] + ".yo";
                        String inputFilePath = problem.testCasesDirPath + testCase.inputFileName;
                        String outputFilePath = basePath + testCase.outputFileName;
                        String desiredOutputFilePath = problem.testCasesDirPath + testCase.outputFileName;

                        executer.execute(targetFilePath, inputFilePath, outputFilePath);

                        String output = new String(Files.readAllBytes(Paths.get(outputFilePath)));
                        String desiredOutput = new String(Files.readAllBytes(Paths.get(desiredOutputFilePath)));

                        if (output.equals(desiredOutput)) {
                            score = testCase.score;
                        } else {
                            score = 0;
                        }

                        problemStudentGrades.add(score);
                    }

                    studentGrades.put(problem.id, problemStudentGrades);

                } catch (Exception ignored) {}
            }

            grades.put(student.id, studentGrades);
        }

        return grades;
    }

    public Map<String,Map<String, List<Double>>> gradeRobust(ExamSpec examSpec, String submissionDirPath) {
        List<Problem> problems = examSpec.problems;
        List<Student> students = examSpec.students;

        Map<String, Map<String, List<Double>>> grades = new HashMap<>();

        for (Student student : students) {
            Map<String, List<Double>> studentGrades = new HashMap<>();

            String studentPath = submissionDirPath + student.id;

            Path defaultPath = Paths.get(submissionDirPath + student.id + "/");
            if (!Files.exists(defaultPath)) {
                File dir = new File(submissionDirPath);
                String[] paths = dir.list();
                if (paths != null) {
                    for (String path : paths) {
                        if (path.contains(student.id)) {
                            studentPath = submissionDirPath + path;
                            break;
                        }
                    }
                }
            }

            for (Problem problem : problems) {
                List<Double> problemStudentGrades = new LinkedList<>();
                List<TestCase> testCases = problem.testCases;

                double deduct = 1.0;

                try {
                    String basePath = studentPath + "/" + problem.id + "/";

                    if (problem.wrappersDirPath != null) {
                        File wrappersDir = new File(problem.wrappersDirPath);
                        String[] wrappersFiles = wrappersDir.list();
                        if (wrappersFiles != null) {
                            for (String wrappersFile : wrappersFiles) {
                                if (wrappersFile.contains(".sugo")) {
                                    try {
                                        Path beforePath = Paths.get(problem.wrappersDirPath + wrappersFile);
                                        Path afterPath = Paths.get(basePath + wrappersFile);
                                        Files.copy(beforePath, afterPath);
                                    } catch (IOException e) {
                                        throw new FileSystemRelatedException();
                                    }
                                }
                            }
                        }
                    }

                    File baseDir = new File(basePath);
                    String[] allFiles = baseDir.list();
                    if (allFiles == null) {
                        throw new FileSystemRelatedException();
                    }
                    for (String fileOrDir : allFiles) {
                        File tempFileOrDir = new File(basePath + fileOrDir);
                        if (tempFileOrDir.isDirectory()) {
                            String[] addPaths = tempFileOrDir.list();
                            if (addPaths != null) {
                                for (String addPath : addPaths) {
                                    try {
                                        Path beforePath = Paths.get(basePath + fileOrDir + "/" + addPath);
                                        Path afterPath = Paths.get(basePath + addPath);
                                        File prevFile = afterPath.toFile();
                                        if(prevFile.isFile()){
                                            Files.delete(afterPath);
                                        }
                                        Files.move(beforePath, afterPath);
                                    } catch (IOException e) {
                                        throw new FileSystemRelatedException();
                                    }
                                }
                            }
                            break;
                        }
                    }

                    String[] filePaths = baseDir.list();
                    if (filePaths == null) {
                        throw new FileSystemRelatedException();
                    }
                    for (String filePath : filePaths) {
                        if (filePath.contains(".yo") && !Arrays.asList(filePaths).contains(filePath.split("\\.")[0] + ".sugo")) {
                            deduct = 0.5;
                        }
                        if (filePath.contains(".sugo")) {
                            compiler.compile(basePath + filePath);
                        }
                    }

                    for (TestCase testCase : testCases) {
                        double score;

                        try {
                            String targetFilePath = basePath + problem.targetFileName.split("\\.")[0] + ".yo";
                            String inputFilePath = problem.testCasesDirPath + testCase.inputFileName;
                            String outputFilePath = basePath + testCase.outputFileName;
                            String desiredOutputFilePath = problem.testCasesDirPath + testCase.outputFileName;

                            executer.execute(targetFilePath, inputFilePath, outputFilePath);

                            String output = new String(Files.readAllBytes(Paths.get(outputFilePath)));
                            String desiredOutput = new String(Files.readAllBytes(Paths.get(desiredOutputFilePath)));

                            if (problem.judgingTypes == null) {
                                problem.judgingTypes = new HashSet<>();
                            }

                            if (problem.judgingTypes.contains(Problem.LEADING_WHITESPACES)) {
                                output = output.replaceAll("^\\s+", "");
                                desiredOutput = desiredOutput.replaceAll("^\\s+", "");
                            }

                            if (problem.judgingTypes.contains(Problem.IGNORE_WHITESPACES)) {
                                output = output.replaceAll("\\s+", "");
                                desiredOutput = desiredOutput.replaceAll("\\s+", "");
                            }

                            if (problem.judgingTypes.contains(Problem.CASE_INSENSITIVE)) {
                                output = output.toLowerCase();
                                desiredOutput = desiredOutput.toLowerCase();
                            }

                            if (problem.judgingTypes.contains(Problem.IGNORE_SPECIAL_CHARACTERS)) {
                                output = output.replaceAll("[^a-zA-Z0-9\\s]", "");
                                desiredOutput = desiredOutput.replaceAll("[^a-zA-Z0-9\\s]", "");
                            }

                            if (output.equals(desiredOutput)) {
                                score = testCase.score * deduct;
                            } else {
                                score = 0.0;
                            }
                        } catch (IOException e) {
                            throw new FileSystemRelatedException();
                        } catch (RunTimeErrorException | InvalidFileTypeException | FileSystemRelatedException e) {
                            score = 0.0;
                        }

                        problemStudentGrades.add(score);
                    }
                } catch (CompileErrorException | InvalidFileTypeException | FileSystemRelatedException e) {
                    for (TestCase testCase : testCases) {
                        problemStudentGrades.add(0.0);
                    }
                }
                studentGrades.put(problem.id, problemStudentGrades);
            }

            grades.put(student.id, studentGrades);
        }

        return grades;
    }
}

