from sqlalchemy import DateTime

class SentimentRecord:
    """
    Represents a sentiment score record.

    This class provides a way to store and retrieve information about a sentiment score entry,
    including the datetime and sentiment score.

    Attributes:
        datetime_t (DateTime): The datetime associated with the sentiment score record.
        sentiment_score (float): The sentiment score value associated with the record.

    Methods:
        json(): Returns the sentiment score record as a JSON-compatible dictionary.
        __repr__(): Returns a string representation of the sentiment score record.
        __eq__(other): Compares two SentimentRecord objects based on their datetime_t.
    """
    def __init__(self, datetime_t: DateTime, sentiment_score: float):
        self.datetime_t = datetime_t
        self.sentiment_score = sentiment_score

    def json(self):
        """
        Return the sentiment score record as a JSON-compatible dictionary.

        Returns:
            dict: A dictionary representing the sentiment score record.
        """
        return {
            'datetime_t': self.datetime_t,
            'sentiment_score': self.sentiment_score
        }

    def __repr__(self):
        """
        Return a string representation of the sentiment score record.

        Returns:
            str: A string representation of the sentiment score record.
        """
        return f"<Record {self.sentiment_score}>"

    def __eq__(self, other):
        """
        Compare two SentimentRecord objects based on their datetime_t.

        Returns:
            bool: True if both objects have the same datetime_t, False otherwise.
        """
        if not isinstance(other, SentimentRecord):
            return False
        return other.datetime_t == self.datetime_t
