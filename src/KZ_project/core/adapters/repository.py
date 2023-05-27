import abc


class AbstractBaseRepository(abc.ABC):
    """
    Abstract base class for repository implementations.

    This class defines the interface for repository implementations and provides common methods.

    Attributes:
        None
    """

    @abc.abstractmethod
    def list(self):
        """
        List all entities in the repository.

        This method retrieves all entities from the repository.

        Returns:
            A list of all entities in the repository.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
