import abc

from pandas import DataFrame
from KZ_project.core.adapters.repository import AbstractBaseRepository
from KZ_project.core.domain.sentiment_record import SentimentRecord

from sqlalchemy import desc, asc


class AbstractSentimentRecordRepository(AbstractBaseRepository):

    @abc.abstractmethod
    def add(self, sentiment_record: SentimentRecord):

        raise NotImplementedError

    @abc.abstractmethod
    def get(self) -> SentimentRecord:

        raise NotImplementedError

    @abc.abstractmethod
    def list(self):

        raise NotImplementedError


class SentimentRecordRepository(AbstractSentimentRecordRepository):

    def __init__(self, session):
        self.session = session

    def add(self, sentiment_record):
        self.session.add(sentiment_record)

    def get(self):
        return self.session.query(SentimentRecord).order_by(desc(SentimentRecord.datetime_t)).first()

    def list(self):
        """
        List all entities in the repository.

        This method retrieves all crypto entities from the database session.

        Returns:
            list: A list of all crypto entities in the repository.
        """
        return self.session.query(SentimentRecord).all()
    
    def add_list_from_dataframe(self, dataframe: DataFrame):
        """
        Add a list of sentiment records to the repository from a DataFrame.

        Args:
            dataframe (pandas.DataFrame): The DataFrame containing sentiment score data.
        """
        sentiment_records_list = []

        for index, row in dataframe.iterrows():
            datetime_t = index.to_pydatetime()  # Convert the DataFrame index to a Python datetime object
            sentiment_score = row['sentiment_score']
            sentiment_record = SentimentRecord(datetime_t=datetime_t, sentiment_score=sentiment_score)
            sentiment_records_list.append(sentiment_record)

        self.session.add_all(sentiment_records_list)