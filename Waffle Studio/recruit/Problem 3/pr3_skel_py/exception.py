from abc import ABCMeta, abstractmethod

class AppException(ABCMeta):

    @abstractmethod
    def __eq__(self, other):
        pass