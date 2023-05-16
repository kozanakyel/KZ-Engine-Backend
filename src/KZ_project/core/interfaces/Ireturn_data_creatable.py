import abc


class IReturnDataCreatable(abc.ABC):
    @abc.abstractmethod
    def create_retuns_data(self, X_pd, y_pred):
        raise NotImplementedError