import abc


class AbstractForecaster(abc.ABC):
    """
    An abstract base class for implementing forecasting models.

    This class defines the interface for creating, training, saving, loading, and evaluating forecasting models.
    Subclasses must implement the abstract methods to provide specific functionality for each method.
    """

    @abc.abstractmethod
    def create_train_test_data(self, x, y, test_size: float):
        """
        Split the input data into training and testing sets.

        Parameters:
            x (array-like): The input features.
            y (array-like): The target variable.
            test_size (float): The proportion of the data to be used for testing.

        Returns:
            tuple: A tuple containing the training and testing data as (X_train, X_test, y_train, y_test).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self):
        """
        Train the forecasting model.

        This method should be implemented to fit the model to the training data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, file_name: str):
        """
        Save the trained model to a file.

        Parameters:
            file_name (str): The name or path of the file to save the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, file_name: str):
        """
        Load a trained model from a file.

        Parameters:
            file_name (str): The name or path of the file to load the model from.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_score(self):
        """
        Evaluate the performance of the forecasting model.

        Returns:
            float: The score or metric representing the performance of the model.
        """
        raise NotImplementedError
