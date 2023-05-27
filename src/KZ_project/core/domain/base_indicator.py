import abc


class BaseIndicator(abc.ABC):
    """
    Abstract base class for creating custom indicator classes.

    Methods:
        create_ind_cols(): Abstract method to create indicator columns.
        create_ichmiouk_tenkansen(dft): Abstract method to create Ichimoku Tenkan-sen indicator column.
        create_ichmiouk_kijunsen(dft): Abstract method to create Ichimoku Kijun-sen indicator column.
    """

    @abc.abstractmethod
    def create_ind_cols(self) -> None:
        """
        Abstract method to be implemented by subclasses for creating indicator columns.

        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_ichmiouk_tenkansen(self, dft) -> None:
        """
        Abstract method to be implemented by subclasses for creating the Ichimoku Tenkan-sen indicator column.

        Args:
            dft (pd.DataFrame): The DataFrame on which the indicator will be calculated.

        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_ichmiouk_kijunsen(self, dft) -> None:
        """
        Abstract method to be implemented by subclasses for creating the Ichimoku Kijun-sen indicator column.

        Args:
            dft (pd.DataFrame): The DataFrame on which the indicator will be calculated.

        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        """
        raise NotImplementedError
