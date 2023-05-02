import abc


class IServiceClient(abc.ABC):
    
    @abc.abstractmethod
    def get_history(self, symbol, interval, start, end = None, *args):
        raise NotImplementedError 