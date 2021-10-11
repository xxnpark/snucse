#!/usr/bin/env python
# -*- coding: utf8 -*-

from request import Request
from exception import AppException
import sys


def read_file(test_mode):
    f = None
    try:
        if test_mode:
            print("helloWorld")
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
            print("success")
            request = Request(line)
            execute(request)
        except AppException as e:
            print(e)


# TODO: 파싱된 request 객체를 받아서 실제 동작을 실행하는 함수입니다.
# @param request 실행할 요청 객체
def execute(request):
    pass


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
    main()
