import abc


class BaseIndicator(abc.ABC):


    @abc.abstractmethod
    def create_ind_cols(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def create_ichmiouk_tenkansen(self, dft) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def create_ichmiouk_kijunsen(self, dft) -> None:
        raise NotImplementedError
