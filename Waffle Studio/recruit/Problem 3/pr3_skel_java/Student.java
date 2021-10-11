//< @brief 학생 클래스입니다.
public class Student {
    // 학년과 이름은 변하지 않기 때문에 final 변수로 선언, getter만 제공
    // pin은 계속 변할 수 있기 때문에 setter도 제공
    final private int grade;
    final private String name;
    private boolean pin;

    // 생성자
    Student(String grade, String name) {
        this.grade = Integer.parseInt(grade);
        this.name = name;
    }

    public int getGrade() {
        return grade;
    }

    public String getName() {
        return name;
    }

    public boolean getPin() {
        return pin;
    }

    public void setPin(boolean pin) {
        this.pin = pin;
    }

    // equals를 override하여 학년과 이름이 같으면 동일한 객체로 취급하도록 함
    @Override
    public boolean equals(Object o) {
        if (o instanceof Student) {
            return ((Student) o).grade == this.grade && ((Student) o).name.equals(this.name);
        }
        return false;
    }

    // toString을 override하여 출력 형식 맞추기 편하도록 함
    @Override
    public String toString() {
        return String.format("%d | %s", this.grade, this.name);
    }
}
