from sqlalchemy import DateTime


class SentimentRecord:

    def __init__(self, datetime_t: DateTime, sentiment_score: float):
        self.datetime_t = datetime_t
        self.sentiment_score = sentiment_score

    def __eq__(self, other):
        if not isinstance(other, SentimentRecord):
            return False
        return other.datetime_t == self.datetime_t

    def json(self):

        return {
            'datetime_t': self.datetime_t,
            'sentiment_score': self.sentiment_score
        }

    def __repr__(self):

        return f"<Record {self.sentiment_score}>"
