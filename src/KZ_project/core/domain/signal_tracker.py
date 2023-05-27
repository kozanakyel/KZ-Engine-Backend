from KZ_project.core.domain.forecast_model import ForecastModel


class SignalTracker:
    """
    Represents a signal tracker that tracks signals, tweet counts, and forecast models for a specific ticker.

    This class provides a way to store and retrieve information about a signal, tweet counts, datetime, backtest returns
    data, and forecast model associated with a particular ticker.

    Attributes:
        signal (int): The signal value associated with the tracker.
        ticker (str): The ticker symbol associated with the tracker.
        tweet_counts (int): The number of tweets associated with the tracker.
        datetime_t (str): The datetime associated with the tracker.
        backtest_returns_data (str): The backtest returns data associated with the tracker.
        forecast_model (ForecastModel): The forecast model associated with the tracker.

    Methods:
        json(): Returns the signal tracker as a JSON-compatible dictionary.
    """
    def __init__(
            self,
            signal: int,
            ticker: str,
            tweet_counts: int,
            datetime_t: str,
            backtest_returns_data: str,
            forecast_model: ForecastModel
    ):
        self.signal = signal
        self.datetime_t = datetime_t
        self.tweet_counts = tweet_counts
        self.ticker = ticker
        self.backtest_returns_data = backtest_returns_data
        self.forecast_model = forecast_model

    def json(self):
        """
        Return the signal tracker as a JSON-compatible dictionary.

        Returns:
            dict: A dictionary representing the signal tracker.
        """
        return {
            'signal': self.signal,
            'datetime_t': self.datetime_t,
            'tweet_counts': self.tweet_counts,
            'ticker': self.ticker,
            'backtest_returns_data': self.backtest_returns_data,
            'forecast_model': self.forecast_model.json()
        }

    def __repr__(self):
        """
        Return a string representation of the signal tracker.

        Returns:
            str: A string representation of the signal tracker.
        """
        return f"<SignalTracker {self.ticker}, datetime: {self.datetime_t}, position: {self.signal}>"
