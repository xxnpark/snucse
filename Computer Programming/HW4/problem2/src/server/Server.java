package server;

import course.*;
import user.User;
import utils.Config;
import utils.ErrorCode;
import utils.Pair;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class Server {
    private Map<Integer, Course> allCourses = new HashMap<>();
    private Map<String, User> users = new HashMap<>();

    private void initializeCourses() {
        allCourses = new HashMap<>();

        String coursePath = "data/Courses/2020_Spring/";
        File courseDir = new File(coursePath);
        String[] colleges = courseDir.list();

        if (colleges == null) {
            return;
        }

        for (String college : colleges) {
            File deptDir = new File(coursePath + college);
            String[] courseIdStrs = deptDir.list();

            if (courseIdStrs == null) {
                continue;
            }

            for (String courseIdStr : courseIdStrs) {
                String courseInfo;

                try {
                    Scanner sc = new Scanner(new File(coursePath + college + "/" + courseIdStr));
                    courseInfo = sc.nextLine();
                } catch (FileNotFoundException e) {
                    continue;
                }

                String[] courseInfoArray = courseInfo.split("\\|");

                int courseId = Integer.parseInt(courseIdStr.split("\\.")[0]);
                String department = courseInfoArray[0];
                String academicDegree = courseInfoArray[1];
                int academicYear = Integer.parseInt(courseInfoArray[2]);
                String courseName = courseInfoArray[3];
                int credit = Integer.parseInt(courseInfoArray[4]);
                String location = courseInfoArray[5];
                String instructor = courseInfoArray[6];
                int quota = Integer.parseInt(courseInfoArray[7]);

                allCourses.put(courseId, new Course(courseId, college, department, academicDegree, academicYear,
                        courseName, credit, location, instructor, quota));
            }
        }
    }

    private boolean initializeUsers() {
        boolean ret = true;
        users = new HashMap<>();
        String userPath = "data/Users/";
        File userDir = new File(userPath);
        String[] userIds = userDir.list();

        if (userIds == null) {
            return false;
        }

        for (String userId : userIds) {
            User user = new User(userId, 0, 0);
            try {
                Scanner sc = new Scanner(new File(userPath + userId + "/bid.txt"));
                while (sc.hasNext()) {
                    String nextLine = sc.nextLine();
                    if (nextLine.equals("")) {
                        continue;
                    }
                    String[] bidStr = nextLine.split("\\|");
                    user.addBidding(Integer.parseInt(bidStr[0]), Integer.parseInt(bidStr[1]));
                }
                users.put(userId, user);
            } catch (IOException e) {
                if (!userId.equals(".DS_Store")) {
                    ret = false;
                }
            }
        }
        return ret;
    }

    public List<Course> search(Map<String,Object> searchConditions, String sortCriteria){
        initializeCourses();
        initializeUsers();

        List<Course> searchedCourses = new LinkedList<>(allCourses.values());

        String dept = null;
        Integer ay = null;
        String name = null;

        if (searchConditions != null) {
            dept = (String) searchConditions.get("dept");
            ay = (Integer) searchConditions.get("ay");
            name = (String) searchConditions.get("name");
        }

        if (dept != null && !dept.equals("")) {
            List<Course> tempCourses = new LinkedList<>();
            for (Course course : searchedCourses) {
                if (course.department.equals(dept)) {
                    tempCourses.add(course);
                }
            }
            searchedCourses = tempCourses;
        }

        if (ay != null) {
            List<Course> tempCourses = new LinkedList<>();
            for (Course course : searchedCourses) {
                if (course.academicYear <= ay) {
                    tempCourses.add(course);
                }
            }
            searchedCourses = tempCourses;
        }

        if (name != null) {
            if (name.equals("")) {
                searchedCourses = new LinkedList<>();
            } else {
                List<Course> tempCourses = new LinkedList<>();
                for (Course course : searchedCourses) {
                    String[] names = name.split(" ");
                    List<String> courseNames = Arrays.asList(course.courseName.split(" "));
                    boolean contains = true;
                    for (String singleName : names) {
                        if (!courseNames.contains(singleName)) {
                            contains = false;
                            break;
                        }
                    }
                    if (contains) {
                        tempCourses.add(course);
                    }
                }
                searchedCourses = tempCourses;
            }
        }

        searchedCourses.sort(new IdComp());

        if (sortCriteria != null) {
            switch (sortCriteria) {
                case "name":
                    searchedCourses.sort(new NameComp());
                    break;
                case "dept":
                    searchedCourses.sort(new DeptComp());
                    break;
                case "ay":
                    searchedCourses.sort(new AyComp());
                    break;
            }
        }

        return searchedCourses;
    }

    public int bid(int courseId, int mileage, String userId){
        initializeCourses();
        initializeUsers();

        if (userId == null) {
            return ErrorCode.USERID_NOT_FOUND;
        }

        User user = users.get(userId);
        boolean userPathExists = Files.exists(Paths.get("data/Users/" + userId));

        if (user == null && !userPathExists) {
            return ErrorCode.USERID_NOT_FOUND;
        }

        if (allCourses.get(courseId) == null) {
            return ErrorCode.NO_COURSE_ID;
        }

        if (mileage < 0) {
            return ErrorCode.NEGATIVE_MILEAGE;
        }

        if (mileage > Config.MAX_MILEAGE_PER_COURSE) {
            return ErrorCode.OVER_MAX_COURSE_MILEAGE;
        }

        if (user == null) {
            return ErrorCode.IO_ERROR;
        }

        int errorCode = user.addBidding(courseId, mileage);

        if (errorCode == ErrorCode.SUCCESS) {
            if (mileage != 0) {
                try {
                    FileWriter fileWriter = new FileWriter("data/Users/" + userId + "/bid.txt", true);
                    fileWriter.append("\n" + courseId + "|" + mileage);
                    fileWriter.flush();
                    fileWriter.close();
                } catch (IOException e) {
                    errorCode = ErrorCode.IO_ERROR;
                }
            }
        } else if (errorCode == ErrorCode.SUCCESS + 1) {
            try {
                String tempPathStr = "data/Users/" + userId + "/temp.txt";
                String bidPathStr = "data/Users/" + userId + "/bid.txt";

                File tempBidFile = new File(tempPathStr);
                tempBidFile.createNewFile();
                Scanner sc = new Scanner(new File(bidPathStr));
                FileWriter fileWriter = new FileWriter(tempPathStr, true);
                while (sc.hasNext()) {
                    String nextLine = sc.nextLine();
                    if (!nextLine.contains(courseId + "|")) {
                        fileWriter.append(nextLine + "\n");
                        fileWriter.flush();
                    } else if (mileage != 0) {
                        fileWriter.append(courseId + "|" + mileage + "\n");
                        fileWriter.flush();
                    }
                }
                fileWriter.close();
                Path beforePath = Paths.get(tempPathStr);
                Path afterPath = Paths.get(bidPathStr);
                File prevFile = afterPath.toFile();
                if (prevFile.isFile()){
                    Files.delete(afterPath);
                }
                Files.move(beforePath, afterPath);
            } catch (IOException ignored) {}
            errorCode = ErrorCode.SUCCESS;
        }

        return errorCode;
    }

    public Pair<Integer,List<Bidding>> retrieveBids(String userId){
        initializeCourses();
        initializeUsers();

        int errorCode;
        List<Bidding> biddings = new LinkedList<>();

        if (userId == null) {
            return new Pair<>(ErrorCode.USERID_NOT_FOUND, biddings);
        }

        String userPathStr = "data/Users/" + userId + "/";
        Path userPath = Paths.get(userPathStr);
        if (!Files.exists(userPath)) {
            errorCode = ErrorCode.USERID_NOT_FOUND;
        } else {
            try {
                Scanner sc = new Scanner(new File(userPathStr + "bid.txt"));
                while (sc.hasNext()) {
                    String nextLine = sc.nextLine();
                    if (nextLine.equals("")) {
                        continue;
                    }
                    String[] bidStr = nextLine.split("\\|");
                    biddings.add(new Bidding(Integer.parseInt(bidStr[0]), Integer.parseInt(bidStr[1])));
                }
                errorCode = ErrorCode.SUCCESS;
            } catch (FileNotFoundException e) {
                errorCode = ErrorCode.IO_ERROR;
            }
        }

        return new Pair<>(errorCode, biddings);
    }

    public boolean confirmBids(){
        initializeCourses();
        boolean bool = initializeUsers();

        List<Course> courseList = new LinkedList<>(allCourses.values());
        List<User> userList = new LinkedList<>(users.values());
        for (Course course : courseList) {
            List<Pair<User, Integer>> courseUserBidList = new LinkedList<>();
            for (User user : userList) {
                Bidding bidding = user.biddings.get(course.courseId);
                if (bidding != null) {
                    courseUserBidList.add(new Pair<>(user, bidding.mileage));
                }
            }

            courseUserBidList.sort(new UserIdComp());
            courseUserBidList.sort(new TotalMileageComp());
            courseUserBidList.sort(new BidComp());

            for (int i = 0; i < course.quota; i++) {
                try {
                    User user = courseUserBidList.get(i).key;

                    String tempPathStr = "data/Users/" + user.userId + "/temp.txt";
                    String bidPathStr = "data/Users/" + user.userId + "/bid.txt";

                    File tempBidFile = new File(tempPathStr);
                    tempBidFile.createNewFile();
                    Scanner sc = new Scanner(new File(bidPathStr));
                    FileWriter fileWriter = new FileWriter(tempPathStr, true);
                    while (sc.hasNext()) {
                        String nextLine = sc.nextLine();
                        if (!nextLine.contains(course.courseId + "|")) {
                            fileWriter.append(nextLine + "\n");
                            fileWriter.flush();
                        }
                    }
                    fileWriter.close();
                    Path beforePath = Paths.get(tempPathStr);
                    Path afterPath = Paths.get(bidPathStr);
                    File prevFile = afterPath.toFile();
                    if (prevFile.isFile()){
                        Files.delete(afterPath);
                    }
                    Files.move(beforePath, afterPath);

                    user.biddings.remove(course.courseId);

                    String userCoursesPathStr = "data/Users/" + user.userId + "/courses.txt";
                    Path userCoursesPath = Paths.get(userCoursesPathStr);
                    if (!Files.exists(userCoursesPath)) {
                        fileWriter = new FileWriter(userCoursesPathStr);
                        fileWriter.write(course.courseId + "\n");
                        fileWriter.flush();
                        fileWriter.close();
                    } else {
                        fileWriter = new FileWriter(userCoursesPathStr, true);
                        fileWriter.append(course.courseId + "\n");
                        fileWriter.flush();
                        fileWriter.close();
                    }
                } catch (IndexOutOfBoundsException e) {
                    break;
                } catch (IOException e) {
                    return false;
                }
            }
        }
        return bool;
    }

    public Pair<Integer,List<Course>> retrieveRegisteredCourse(String userId){
        if (userId == null) {
            return new Pair<>(ErrorCode.USERID_NOT_FOUND, new LinkedList<>());
        }

        User user = users.get(userId);
        boolean userPathExists = Files.exists(Paths.get("data/Users/" + userId));

        if (user == null) {
            if (userPathExists) {
                return new Pair<>(ErrorCode.IO_ERROR, new LinkedList<>());
            } else {
                return new Pair<>(ErrorCode.USERID_NOT_FOUND, new LinkedList<>());
            }
        }

        List<Course> courses = new LinkedList<>();

        try {
            Scanner sc = new Scanner(new File("data/Users/" + userId + "/courses.txt"));
            while (sc.hasNext()) {
                String nextLine = sc.nextLine();
                if (!nextLine.equals("")) {
                    Course course = allCourses.get(Integer.parseInt(nextLine));
                    courses.add(course);
                }
            }
        } catch (IOException e) {
            return new Pair<>(ErrorCode.SUCCESS, new LinkedList<>());
        }

        return new Pair<>(ErrorCode.SUCCESS, courses);
    }
}


class IdComp implements Comparator<Course> {
    public int compare(Course course1, Course course2) {
        return course1.courseId - course2.courseId;
    }
}


class NameComp implements Comparator<Course> {
    public int compare(Course course1, Course course2) {
        return course1.courseName.compareTo(course2.courseName);
    }
}


class DeptComp implements Comparator<Course> {
    public int compare(Course course1, Course course2) {
        return course1.department.compareTo(course2.department);
    }
}


class AyComp implements Comparator<Course> {
    public int compare(Course course1, Course course2) {
        return course1.academicYear - course2.academicYear;
    }
}


class UserIdComp implements Comparator<Pair<User, Integer>> {
    public int compare(Pair<User, Integer> pair1, Pair<User, Integer> pair2) {
        return pair1.key.userId.compareTo(pair2.key.userId);
    }
}


class TotalMileageComp implements Comparator<Pair<User, Integer>> {
    public int compare(Pair<User, Integer> pair1, Pair<User, Integer> pair2) {
        return pair1.key.getMileageCount() - pair2.key.getMileageCount();
    }
}


class BidComp implements Comparator<Pair<User, Integer>> {
    public int compare(Pair<User, Integer> pair1, Pair<User, Integer> pair2) {
        return pair2.value - pair1.value;
    }
}
