from enum import Enum

class Command(Enum):
    add = 0
    delete = 1
    list = 2
    pin = 3
    unpin = 4
    q = 5

class Option(Enum):
    r = 0
    o = 1
    g = 2
    n = 3
    a = 4

class Request:
    def __init__(self, line):
        linelist = line.split()
        self.command = Command[linelist[0]].value

        # line 커맨드가 입력되었을 경우 옵션 중복이 가능하므로 각 옵션에 따른 정보를 순서에 맞게 필드에 저장
        if self.command == 2:
            self.option = []
            self.data = []
            k = 1
            while True:
                if k >= len(linelist) : break
                try:
                    self.option.append(Option[linelist[k][1:]].value)
                    if Option[linelist[k][1:]].value == 0 : self.data.append("")
                except:
                    self.data.append(linelist[k])
                k += 1

        # q 커맨드가 입력되었을 경우 추가 옵션이 없으므로 pass
        elif self.command == 5 : pass

        # add, delete, pin, unpin 커맨드가 입력되었을 경우 옵션 및 정보를 분리하여 필드에 저장
        else:
            try:
                self.option = Option[linelist[1][1:]].value
                self.data = linelist[2:]
            except KeyError:
                self.option = -1
                self.data = linelist[1:]
            except IndexError:
                self.option = -1
                if type(linelist[1]) == int : self.data = linelist[1:]
                else : self.data = None