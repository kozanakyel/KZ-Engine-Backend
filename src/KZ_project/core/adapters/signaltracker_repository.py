import abc
from KZ_project.core.domain.signal_tracker import SignalTracker
from KZ_project.core.adapters.repository import AbstractBaseRepository

from sqlalchemy import desc, asc


class AbstractSignalTrackerRepository(AbstractBaseRepository):
    @abc.abstractmethod
    def add(self, signal_tracker: SignalTracker):
        """
        Add a Signal Tracker entity to the repository.
        This method adds the given Signal Tracker entity to the repository.

        Args:
            signal_tracker (SignalTracker): The Signal Tracker entity to be added.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, ticker) -> SignalTracker:
        """
        Get a signal tracker by ticker.
        This method retrieves the signal tracker with the specified ticker from the repository.

        Args:
            ticker (str): The ticker symbol of the signal tracker.

        Returns:
            SignalTracker: The signal tracker with the specified ticker.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list(self):
        """
        List all entities in the repository.

        This method retrieves all entities from the repository.

        Returns:
            list: A list of all entities in the repository.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class SignalTrackerRepository(AbstractSignalTrackerRepository):
    def __init__(self, session):
        self.session = session

    def add(self, signal_tracker):
        """
        Add a signal tracker to the repository.

        This method adds the specified signal tracker to the repository.

        Args:
            signal_tracker (SignalTracker): The signal tracker to add.

        Returns:
            None

        Raises:
            None
        """
        self.session.add(signal_tracker)

    def get(self, forecast_model_id):
        """
        Get a signal tracker by forecast model ID.

        This method retrieves the signal tracker with the specified forecast model ID from the repository.

        Args:
            forecast_model_id (int): The ID of the forecast model.

        Returns:
            SignalTracker: The signal tracker with the specified forecast model ID.

        Raises:
            None
        """
        return self.session.query(SignalTracker).filter_by(forecast_model_id=forecast_model_id).order_by(
            desc(SignalTracker.datetime_t)).first()

    def list(self):
        """
        List all signal trackers in the repository.

        This method retrieves all signal trackers from the repository.

        Returns:
            list: A list of all signal trackers in the repository.

        Raises:
            None
        """
        return self.session.query(SignalTracker).all()
    
    def list_with_ticker(self, ticker: str):
        return self.session.query(SignalTracker).filter_by(ticker=ticker).all()
