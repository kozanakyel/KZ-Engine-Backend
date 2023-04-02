import abc
from KZ_project.domain.asset import Asset
from KZ_project.Infrastructure.adapters.repository import AbstractBaseRepository

class AbstractAssetRepository(AbstractBaseRepository):
    @abc.abstractmethod
    def add(self, asset: Asset):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, symbol) -> Asset:
        raise NotImplementedError
    
class AssetRepository(AbstractAssetRepository):
    def __init__(self, session):
        self.session = session

    def add(self, asset):
        self.session.add(asset)

    def get(self, symbol):
        return self.session.query(Asset).filter_by(symbol=symbol).first()
    
    def list(self):
        return self.session.query(Asset).all()