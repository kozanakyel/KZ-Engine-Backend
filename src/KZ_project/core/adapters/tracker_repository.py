import abc
from KZ_project.core.domain.tracker import Tracker
from KZ_project.core.adapters.repository import AbstractBaseRepository

from sqlalchemy import desc

class AbstractTrackerRepository(AbstractBaseRepository):
    @abc.abstractmethod
    def add(self, tracker: Tracker):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, symbol) -> Tracker:
        raise NotImplementedError
    
class TrackerRepository(AbstractTrackerRepository):
    def __init__(self, session):
        self.session = session

    def add(self, tracker):
        self.session.add(tracker)

    def get(self, symbol):
        return self.session.query(Tracker).filter_by(symbol=symbol).order_by(desc(Tracker.created_at)).first()
    
    def list(self):
        return self.session.query(Tracker).all()