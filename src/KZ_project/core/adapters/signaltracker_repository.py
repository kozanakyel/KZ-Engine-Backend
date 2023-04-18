import abc
from KZ_project.core.domain.signal_tracker import SignalTracker
from KZ_project.core.adapters.repository import AbstractBaseRepository

from sqlalchemy import desc

class AbstractSignalTrackerRepository(AbstractBaseRepository):
    @abc.abstractmethod
    def add(self, signal_tracker: SignalTracker):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, ticker) -> SignalTracker:
        raise NotImplementedError
    
class SignalTrackerRepository(AbstractSignalTrackerRepository):
    def __init__(self, session):
        self.session = session

    def add(self, signal_tracker):
        self.session.add(signal_tracker)

    def get(self, ticker):
        # why datetime_t and not created_at, we investigate them !!!
        return self.session.query(SignalTracker).filter_by(ticker=ticker).order_by(desc(SignalTracker.datetime_t)).first()
    
    def list(self):
        return self.session.query(SignalTracker).all()