import abc


class BaseIndicator(abc.ABC):
    """
    Do the all indicators methods abstract Dont Forget!!!
    """
    
    @abc.abstractmethod
    def create_ind_cols(self) -> None:
        raise NotImplementedError