import abc


class BaseIndicator(abc.ABC):
    
    @abc.abstractmethod
    def create_ind_cols(self) -> None:
        raise NotImplementedError