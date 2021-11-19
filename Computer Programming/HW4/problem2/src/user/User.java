package user;

import course.Bidding;
import utils.Config;
import utils.ErrorCode;

import java.util.*;

public class User {
    public final String userId;
    private int mileageCount;
    private int courseCount;
    public Map<Integer, Bidding> biddings = new HashMap<>();

    public User(String userId, int mileageCount, int courseCount){
        this.userId = userId;
        this.mileageCount = mileageCount;
        this.courseCount = courseCount;
    }

    public int addBidding(int courseId, int mileage) {
        Bidding bidding = biddings.get(courseId);
        if (bidding != null) {
            if (mileageCount - bidding.mileage + mileage > Config.MAX_MILEAGE) {
                return ErrorCode.OVER_MAX_MILEAGE;
            }
            if (mileage == 0) {
                biddings.remove(courseId);
                mileageCount = mileageCount - bidding.mileage;
                courseCount--;
            } else {
                mileageCount = mileageCount - bidding.mileage + mileage;
                bidding.mileage = mileage;
            }
            return ErrorCode.SUCCESS + 1;
        } else {
            if (mileage != 0) {
                if (mileageCount + mileage > Config.MAX_MILEAGE) {
                    return ErrorCode.OVER_MAX_MILEAGE;
                } else if (courseCount == Config.MAX_COURSE_NUMBER) {
                    return ErrorCode.OVER_MAX_COURSE_NUMBER;
                }
                biddings.put(courseId, new Bidding(courseId, mileage));
                mileageCount += mileage;
                courseCount++;
            }
            return ErrorCode.SUCCESS;
        }
    }

    public int getCourseCount() {
        return courseCount;
    }

    public int getMileageCount() {
        return mileageCount;
    }
}
