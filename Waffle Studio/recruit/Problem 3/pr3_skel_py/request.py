from enum import Enum

class Command(Enum):
    ADD = 0
    DELETE = 1
    LIST = 2
    PIN = 3
    UNPIN = 4
    QUIT = 5

class Option(Enum):
    r = 0
    o = 1
    g = 2
    n = 3
    a = 4

class Request:
    # TODO: fields/attributes

    # TODO: constructor
    def __init__(self, command):
        pass