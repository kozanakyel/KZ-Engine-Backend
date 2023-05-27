from KZ_project.core.domain.crypto import Crypto


class ForecastModel:
    """
    Represents a forecast model with various attributes.

    This class provides a convenient way to store and retrieve information about a forecast model, such as its symbol,
    data source, feature counts, model name, interval, AI type, hashtag, accuracy score, datetime, and crypto data.

    Attributes:
        symbol (str): The symbol associated with the forecast model.
        source (str): The data source used for the forecast model.
        feature_counts (int): The number of features used in the model.
        model_name (str): The name of the forecast model.
        interval (str): The time interval used for the forecast.
        ai_type (str): The type of AI used in the forecast model.
        hashtag (str): A hashtag associated with the forecast model.
        accuracy_score (float): The accuracy score of the forecast model.
        datetime_t (str): The datetime associated with the forecast model.
        crypto (Crypto): The crypto data associated with the forecast model.

    Methods:
        json(): Returns the forecast model as a JSON-compatible dictionary.
    """

    def __init__(
            self,
            symbol: str,
            source: str,
            feature_counts: int,
            model_name: str,
            interval: str,
            ai_type: str,
            hashtag: str,
            accuracy_score: float,
            datetime_t: str,
            crypto: Crypto
    ):
        self.symbol = symbol
        self.source = source
        self.feature_counts = feature_counts
        self.model_name = model_name
        self.interval = interval
        self.ai_type = ai_type
        self.hashtag = hashtag
        self.accuracy_score = accuracy_score
        self.datetime_t = datetime_t
        self.crypto = crypto

    def json(self):
        """
        Return the forecast model as a JSON-compatible dictionary.

        Returns:
            dict: A dictionary representing the forecast model.
        """
        return {
            'symbol': self.symbol,
            'source': self.source,
            'feature_counts': self.feature_counts,
            'model_name': self.model_name,
            'interval': self.interval,
            'ai_type': self.ai_type,
            'hashtag': self.hashtag,
            'accuracy_score': self.accuracy_score,
            'datetime_t': self.datetime_t,
            'crypto': self.crypto.json()
        }

    def __eq__(self, other):
        """
        Compare if two forecast models are equal.

        Parameters:
            other (ForecastModel): The other forecast model to compare.

        Returns:
            bool: True if the forecast models are equal, False otherwise.
        """
        if not isinstance(other, ForecastModel):
            return False
        c1 = str(other) == str(self)
        return c1

    def __repr__(self):
        """
        Return a string representation of the forecast model.

        Returns:
            str: A string representation of the forecast model.
        """
        return f"<ForecastModel {self.model_name}, symbol: {self.symbol}, interval: {self.interval}, aitype: {self.ai_type}, datetime_t: {self.datetime_t}>"
