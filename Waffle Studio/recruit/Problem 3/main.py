#!/usr/bin/env python
# -*- coding: utf8 -*-

from request import Request
from exception import AppException
from student import Student
import sys

def read_file(test_mode):
    f = None
    try:
        if test_mode:
            f = open("test-script.txt", "r")
        else:
            f = sys.stdin
    except FileNotFoundError:
        print("File not found.")
        quit()

    while True:
        line = f.readline()
        if line == "":
            break
        try:
            request = Request(line)
            execute(request)
        except Exception as e:
            print(e)

# list -o grade 명령 시 사용될 학년 정렬 함수
# @param stdlist 정렬할 학생들이 포함된 리스트
def sortgrade(stdlist):
    first = []
    second = []
    third = []
    for student in stdlist:
        if student.get_grade() == 1 : first.append(student)
        elif student.get_grade() == 2 : second.append(student)
        elif student.get_grade() == 3 : third.append(student)
    ret = first + second + third
    return ret

# list -o name 명령 시 사용될 이름 정렬 함수
# @param stdlist 정렬할 학생들이 포함된 리스트
def sortname(stdlist):
    names = []
    ret = []
    for i in range(len(stdlist)):
        names.append(str(f"{stdlist[i].get_name():<10}{i*10**(-len(str(len(stdlist)))):.{len(str(len(stdlist)))}f}"))
    names.sort()
    for name in names:
        ret.append(stdlist[int(float(name.split()[1])*10**len(str(len(stdlist))))])
    return ret

# 파싱된 request 객체를 받아서 실제 동작을 실행하는 함수
# @param request 실행할 요청 객체
def execute(request):
    global students

    # add 커맨드 입력 시 학생 중복 여부 확인한 후 추가
    if request.command == 0:
        if request.option == -1:
            student = Student(request.data[0], request.data[1])
            if student not in students : students.append(student)
            else : print("Error 100")
        elif request.option == 4:
            i = 0
            temp = []
            while True:
                try:
                    student = Student(request.data[i], request.data[i+1])
                    if (student not in students) and (student not in temp) : temp.append(student)
                    else:
                        print("Error 100")
                        temp = []
                        break
                except:
                    break
                i += 2
            students += temp

    # delete 커맨드 입력 시 학생 존재 여부 확인 후 제거
    elif request.command == 1:
        student = Student(request.data[0], request.data[1])
        if student in students : students.remove(student)
        else : print("Error 200")

    # list 커맨드 입력 시 pin / unpin 에 해당하는 학생 리스트를 만든 후,
    # 각 옵션이 입력되었을 시 출력 규칙에 맞게 두 리스트를 변경
    elif request.command == 2:
        pin = []
        unpin = []
        for student in students:
            if student.get_pin() : pin.append(student)
            else : unpin.append(student)
        # -r 옵션은 우선적으로 처리
        if 0 in request.option:
            pin.reverse()
            unpin.reverse()
            del request.data[request.option.index(0)]
            request.option.remove(0)
        for i in range(len(request.option)):
            if request.option[i] == 1:
                if request.data[i] == "grade":
                    pin = sortgrade(pin)
                    unpin = sortgrade(unpin)
                elif request.data[i] == "name":
                    pin = sortname(pin)
                    unpin = sortname(unpin)
            elif request.option[i] == 2:
                tpin = []
                tunpin = []
                for student in pin:
                    if student.get_grade() == int(request.data[i]) : tpin.append(student)
                for student in unpin:
                    if student.get_grade() == int(request.data[i]) : tunpin.append(student)
                pin = tpin
                unpin = tunpin
            elif request.option[i] == 3:
                tpin = []
                tunpin = []
                for student in pin:
                    if student.get_name() == request.data[i] : tpin.append(student)
                for student in unpin:
                    if student.get_name() == request.data[i] : tunpin.append(student)
                pin = tpin
                unpin = tunpin
        for student in pin : print(student)
        for student in unpin : print(student)

    # pin 커맨드 입력 시 해당 학생 객체의 pin 필드 변경
    elif request.command == 3:
        student = Student(request.data[0], request.data[1])
        if student not in students : print("Error 200")
        else:
            for std in students:
                if std == student : std.set_pin(True)

    # unpin 커맨드 입력 시 해당 학생 객체의 pin 필드 변경
    elif request.command == 4:
        student = Student(request.data[0], request.data[1])
        if student not in students:
            print("Error 200")
        else:
            for std in students:
                if std == student : std.set_pin(None)

    # q 커맨드 입력 시 종료
    elif request.command == 5:
        quit()

def main():
    test_mode = ""
    if len(sys.argv) > 1:
        test_mode = sys.argv[1]
    if test_mode == "--test":
        read_file(True)
    else:
        read_file(False)


# main
if __name__ == '__main__':
    students = []
    main()
    '''
    students = []
    while True:
        request = Request(input())
        execute(request)
    '''
