from abc import ABCMeta, abstractmethod

class AppException(ABCMeta, Exception):
    @abstractmethod
    def __init__(self):
        pass

class sException(AppException):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
