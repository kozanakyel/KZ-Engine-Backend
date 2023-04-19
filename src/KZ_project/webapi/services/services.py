from __future__ import annotations
from KZ_project.core.adapters.crypto_repository import CryptoRepository
from KZ_project.core.domain.asset import Asset, allocate_tracker
from KZ_project.core.domain.aimodel import AIModel
from KZ_project.core.domain.forecast_model import ForecastModel
from KZ_project.core.domain.signal_tracker import SignalTracker
from KZ_project.core.domain.tracker import Tracker
from KZ_project.core.domain.crypto import Crypto
from KZ_project.core.domain.asset import InvalidSymbol

from KZ_project.core.adapters.repository import AbstractBaseRepository

def add_asset(
    symbol: str, source: str,
    repo: AbstractBaseRepository, session,
) -> None:
    repo.add(Asset(symbol, source))
    session.commit()
    
def is_valid_symbol(symbol, assets):
    return symbol in {asset.symbol for asset in assets}

    
def allocate_tracker_service(
    symbol: str, datetime_t: str, position: int,
    repo: AbstractBaseRepository, session
) -> tuple:
    tracker = Tracker(symbol, datetime_t, position)
    print(f'tracker created_at : {tracker.created_at}')
    assets = repo.list()
    if not is_valid_symbol(tracker.symbol, assets):
        raise InvalidSymbol(f"Invalid symbol {tracker.symbol}")
    result_tracker = allocate_tracker(tracker, assets)
    session.commit()
    print(f'result print allocatie tracker:L {result_tracker}')
    return result_tracker

def get_position(
    symbol: str, 
    repo: AbstractBaseRepository, session,
):
    result = repo.get(symbol)
    session.commit()
    return result


def add_aimodel(
    symbol: str, source: str, feature_counts: int,
    model_name: str, ai_type: str, hashtag: str, accuracy_score: float,
    repo: AbstractBaseRepository, session,
):
    repo.add(AIModel(symbol, source, feature_counts, model_name, 
                     ai_type, hashtag, accuracy_score))
    session.commit()
    
def get_aimodel(
    symbol: str, 
    repo: AbstractBaseRepository, session,
):
    result = repo.get(symbol)
    session.commit()
    return result 


#######################
class InvalidName(Exception):
    pass

def add_crypto(
    name: str, ticker: str, description:str,
    repo: AbstractBaseRepository, session,
) -> None:
    crypto_list = repo.list()
    crypto_name_list = [x.name for x in crypto_list]
    
    if name in crypto_name_list:
        raise InvalidName(f'Error This name is exist: {name}')
    repo.add(Crypto(name, ticker, description))
    session.commit()
    
def get_crypto(
    ticker: str, 
    repo: AbstractBaseRepository, session,
) -> None:
    crypto_list = repo.list()
    crypto_ticker_list = [x.ticker for x in crypto_list]
    
    if ticker not in crypto_ticker_list:
        raise InvalidName(f'Error This ticker is not exist: {ticker}')
    result = repo.get(ticker)
    session.commit()
    return result 

def add_forecast_model(
    symbol:str, source:str, feature_counts:int, model_name:str,
    interval:str, ai_type:str, hashtag:str, accuracy_score:float,
    crypto, repo: AbstractBaseRepository, session
) -> None:
    #repo_cr = CryptoRepository(session)
    #finding_crypto = get_crypto(ticker=hashtag, repo=repo_cr, session=session)
    
    repo.add(ForecastModel(symbol, source, feature_counts, model_name,
                          interval, ai_type, hashtag, accuracy_score, crypto))
    session.commit()
    
def get_forecast_model(
    symbol: str, interval:str, ai_type:str, 
    repo: AbstractBaseRepository, session,
) -> None:
    
    result = repo.get(symbol, interval, ai_type)
    
    session.commit()
    return result 

def add_signal_tracker(
    signal:int, ticker:str,  tweet_counts:int, datetime_t:str,
    forecast_model: ForecastModel,
    repo: AbstractBaseRepository, session
) -> None:
    repo.add(SignalTracker(signal, ticker, tweet_counts, datetime_t,
                          forecast_model))
    session.commit()
    
def get_signal_tracker(
    forecast_model_id:int, 
    repo: AbstractBaseRepository, session,
) -> None:
    
    result = repo.get(forecast_model_id)
    
    session.commit()
    return result 
    

