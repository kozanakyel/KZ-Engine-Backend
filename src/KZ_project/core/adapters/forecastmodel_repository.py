import abc
from KZ_project.core.adapters.repository import AbstractBaseRepository
from KZ_project.core.domain.forecast_model import ForecastModel
from sqlalchemy import desc, func, and_


class AbstractForecastModelRepository(AbstractBaseRepository):
    """
    Abstract base class for forecast model repository implementations.

    This class defines the interface for forecast model repository implementations and provides common methods.

    Attributes:
        None
    """

    @abc.abstractmethod
    def add(self, forecast_model: ForecastModel):
        """
        Add a forecast model entity to the repository.

        This method adds the given forecast model entity to the repository.

        Args:
            forecast_model (ForecastModel): The forecast model entity to be added.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, symbol, interval, ai_type) -> ForecastModel:
        """
        Get a forecast model entity from the repository.

        This method retrieves the forecast model entity with the specified symbol, interval, and AI type from the repository.

        Args:
            symbol (str): The symbol of the forecast model entity to retrieve.
            interval (str): The interval of the forecast model entity to retrieve.
            ai_type (str): The AI type of the forecast model entity to retrieve.

        Returns:
            ForecastModel: The retrieved forecast model entity.

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


class ForecastModelRepository(AbstractForecastModelRepository):
    """
    Implementation of the forecast model repository using a database session.

    This class provides methods to add, get, list, and retrieve last forecast models from a database session.

    Attributes:
        session (object): The database session.
    """

    def __init__(self, session):
        self.session = session

    def add(self, forecast_model: ForecastModel):
        """
        Add a forecast model entity to the repository.

        This method adds the given forecast model entity to the database session.

        Args:
            forecast_model (ForecastModel): The forecast model entity to be added.

        Returns:
            None
        """
        self.session.add(forecast_model)

    def get(self, symbol, interval, ai_type):
        """
        Get a forecast model entity from the repository.

        This method retrieves the forecast model entity with the specified symbol, interval, and AI type from the database session.

        Args:
            symbol (str): The symbol of the forecast model entity to retrieve.
            interval (str): The interval of the forecast model entity to retrieve.
            ai_type (str): The AI type of the forecast model entity to retrieve.

        Returns:
            ForecastModel: The retrieved forecast model entity.
        """
        return self.session.query(ForecastModel) \
            .filter_by(symbol=symbol) \
            .filter_by(interval=interval) \
            .filter_by(ai_type=ai_type) \
            .order_by(desc(ForecastModel.datetime_t)) \
            .first()

    def list(self):
        """
        List all entities in the repository.

        This method retrieves all forecast model entities from the database session.

        Returns:
            list: A list of all forecast model entities in the repository.
        """
        return self.session.query(ForecastModel).all()

    def get_last_forecast_models(self, interval, ai_type):
        """
        Get the last forecast models for each symbol with the specified interval and AI type.

        This method retrieves the last forecast models for each symbol with the specified interval and AI type from
        the database session.

        Args:
            interval (str): The interval of the forecast models to retrieve.
            ai_type (str): The AI type of the forecast models to retrieve.

        Returns:
            list: A list of the last forecast models for each symbol with the specified interval and AI type.

        Raises:
            None
        """
        subq = self.session.query(
            ForecastModel.symbol.label('symbol'),
            func.max(ForecastModel.datetime_t).label('max_datetime_t')
        ).filter_by(interval=interval) \
            .filter_by(ai_type=ai_type) \
            .group_by(ForecastModel.symbol).subquery()

        q = self.session.query(ForecastModel).join(
            subq, and_(
                ForecastModel.symbol == subq.c.symbol,
                ForecastModel.datetime_t == subq.c.max_datetime_t
            )
        )

        return q.all()
