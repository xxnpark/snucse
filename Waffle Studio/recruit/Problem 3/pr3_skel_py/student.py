class Student:

    def __init__(self, grade, name):
        self.grade = int(grade)
        self.name = name
        self.pin = None

    def get_grade(self):
        return self.grade

    def get_name(self):
        return self.name

    def get_pin(self):
        return self.pin

    def set_pin(self, pin):
        self.pin = pin

    # 학년과 이름이 똑같으면 똑같은 객체로 취급함
    def __eq__(self, other):
        if isinstance(other, Student):
            return other.grade == self.grade and other.name == self.name
        return False

    # 출력 형식 맞추기 편하도록 함
    def __str__(self):
        return "{} | {}".format(self.name, self.grade)
