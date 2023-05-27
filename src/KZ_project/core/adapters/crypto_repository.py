import abc
from KZ_project.core.adapters.repository import AbstractBaseRepository
from KZ_project.core.domain.crypto import Crypto


class AbstractCryptoRepository(AbstractBaseRepository):
    """
    Abstract base class for crypto repository implementations.

    This class defines the interface for crypto repository implementations and provides common methods.

    Attributes:
        None
    """

    @abc.abstractmethod
    def add(self, crypto: Crypto):
        """
        Add a crypto entity to the repository.

        This method adds the given crypto entity to the repository.

        Args:
            crypto (Crypto): The crypto entity to be added.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, name) -> Crypto:
        """
        Get a crypto entity from the repository.

        This method retrieves the crypto entity with the specified name from the repository.

        Args:
            name (str): The name of the crypto entity to retrieve.

        Returns:
            Crypto: The retrieved crypto entity.

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


class CryptoRepository(AbstractCryptoRepository):
    """
    Implementation of the crypto repository using a database session.

    This class provides methods to add, get, and list crypto entities from a database session.

    Attributes:
        session (object): The database session.
    """

    def __init__(self, session):
        self.session = session

    def add(self, crypto):
        """
        Add a crypto entity to the repository.

        This method adds the given crypto entity to the database session.

        Args:
            crypto (Crypto): The crypto entity to be added.

        Returns:
            None
        """
        self.session.add(crypto)

    def get(self, ticker):
        """
        Get a crypto entity from the repository.

        This method retrieves the crypto entity with the specified ticker from the database session.

        Args:
            ticker (str): The ticker of the crypto entity to retrieve.

        Returns:
            Crypto: The retrieved crypto entity.
        """
        return self.session.query(Crypto).filter_by(ticker=ticker).first()

    def list(self):
        """
        List all entities in the repository.

        This method retrieves all crypto entities from the database session.

        Returns:
            list: A list of all crypto entities in the repository.
        """
        return self.session.query(Crypto).all()
