import abc
from KZ_project.core.domain.aimodel import AIModel
from KZ_project.core.adapters.repository import AbstractBaseRepository

class AbstractAIModelRepository(AbstractBaseRepository):
    @abc.abstractmethod
    def add(self, aimodel: AIModel):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, symbol) -> AIModel:
        raise NotImplementedError
    
class AIModelRepository(AbstractAIModelRepository):
    def __init__(self, session):
        self.session = session

    def add(self, aimodel):
        self.session.add(aimodel)

    def get(self, symbol):
        return self.session.query(AIModel).filter_by(symbol=symbol).first()
    
    def list(self):
        return self.session.query(AIModel).all()