from sqlalchemy import Table, MetaData, Column, Integer, String, Date, ForeignKey, DateTime, Float
from sqlalchemy.orm import relationship, registry

from KZ_project.core.domain.crypto import Crypto
from KZ_project.core.domain.forecast_model import ForecastModel
from KZ_project.core.domain.sentiment_record import SentimentRecord
from KZ_project.core.domain.signal_tracker import SignalTracker
from KZ_project.core.domain.user import User

mapper_registry = registry()

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("wallet", String(200), primary_key=True),
    Column("username", String(40)),
    Column("email", String(100))
)

cryptos = Table(
    "cryptos",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(100), unique=True),
    Column("ticker", String(25)),
    Column("description", String(500))
)

forecast_models = Table(
    "forecast_models",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(25)),
    Column("source", String(100)),
    Column("feature_counts", Integer),
    Column("model_name", String(500)),
    Column("interval", String(10)),
    Column("ai_type", String(200)),
    Column("hashtag", String(100), nullable=True),
    Column("accuracy_score", Float()),
    Column("datetime_t", String(200)),
    Column("crypto_id", ForeignKey("cryptos.id")),
)

signal_trackers = Table(
    "signal_trackers",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("datetime_t", String(200)),
    Column("signal", Integer, nullable=False),
    Column("ticker", String(25)),
    Column("japanese_candle", String(60)),
    Column("tweet_counts", Integer),
    Column("backtest_returns_data", String()),
    Column("forecast_model_id", ForeignKey("forecast_models.id")),
)

sentiment_records = Table(
    "sentiment_records",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("datetime_t", DateTime, index=True, unique=True),
    Column("sentiment_score", Float)
)


def start_mappers():
    user_mapper = mapper_registry.map_imperatively(User, users)
    crypto_mapper = mapper_registry.map_imperatively(Crypto, cryptos)
    sentiment_mapper = mapper_registry.map_imperatively(SentimentRecord, sentiment_records)
    forecast_model_mapper = mapper_registry.map_imperatively(ForecastModel, forecast_models,
                                                             properties={
                                                                 'crypto': relationship(crypto_mapper)
                                                             })
    mapper_registry.map_imperatively(SignalTracker, signal_trackers,
                                     properties={
                                         'forecast_model': relationship(forecast_model_mapper)
                                     })
