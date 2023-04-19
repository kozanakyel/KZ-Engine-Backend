import abc
from KZ_project.core.adapters.repository import AbstractBaseRepository
from KZ_project.core.domain.crypto import Crypto


class AbstractCryptoRepository(AbstractBaseRepository):
    @abc.abstractmethod
    def add(self, crypto: Crypto):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, name) -> Crypto:
        raise NotImplementedError
    
class CryptoRepository(AbstractCryptoRepository):
    def __init__(self, session):
        self.session = session

    def add(self, crypto):
        self.session.add(crypto)

    def get(self, ticker):
        return self.session.query(Crypto).filter_by(ticker=ticker).first()
    
    def list(self):
        return self.session.query(Crypto).all()