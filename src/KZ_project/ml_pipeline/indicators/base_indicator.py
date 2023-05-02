import abc


class BaseIndicator(abc.ABC):
    
    @abc.abstractmethod
    def create_ind_cols_ta(self) -> None:
        raise NotImplementedError