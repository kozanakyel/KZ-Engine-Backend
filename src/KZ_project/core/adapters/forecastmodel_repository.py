import abc
from KZ_project.core.adapters.repository import AbstractBaseRepository
from KZ_project.core.domain.forecast_model import ForecastModel

from sqlalchemy import desc, func, and_


class AbstractForecastModelRepository(AbstractBaseRepository):
    @abc.abstractmethod
    def add(self, forecast_model: ForecastModel):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, symbol, interval, ai_type) -> ForecastModel:
        raise NotImplementedError
    
class ForecastModelRepository(AbstractForecastModelRepository):
    def __init__(self, session):
        self.session = session

    def add(self, forecast_model):
        self.session.add(forecast_model)
    
    def get(self, symbol, interval, ai_type):
        return self.session.query(ForecastModel)          \
                .filter_by(symbol=symbol)                 \
                .filter_by(interval=interval)             \
                .filter_by(ai_type=ai_type)               \
                .order_by(desc(ForecastModel.datetime_t)) \
                .first()
    
    def list(self):
        return self.session.query(ForecastModel).all()
    
    def get_last_forecast_models(self, interval, ai_type):
        subq = self.session.query(
                    ForecastModel.symbol.label('symbol'),
                    func.max(ForecastModel.datetime_t).label('max_datetime_t')
                ).filter_by(interval=interval)\
                 .filter_by(ai_type=ai_type)\
                 .group_by(ForecastModel.symbol).subquery()

        q = self.session.query(ForecastModel).join(
                subq, and_(
                    ForecastModel.symbol == subq.c.symbol,
                    ForecastModel.datetime_t == subq.c.max_datetime_t
                )
            )

        return q.all()