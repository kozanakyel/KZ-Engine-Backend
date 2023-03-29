import abc


class AbstractBaseRepository():
    @abc.abstractmethod
    def add():
        raise NotImplementedError
    
    @abc.abstractmethod
    def get():
        raise NotImplementedError