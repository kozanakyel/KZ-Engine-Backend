from __future__ import annotations
from KZ_project.core.domain.asset import Asset, allocate_tracker
from KZ_project.core.domain.aimodel import AIModel
from KZ_project.core.domain.tracker import Tracker
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
    return result.symbol, result.position, result.datetime_t


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
    r = repo.get(symbol)
    session.commit()
    return r.symbol, r.source, r.feature_counts, r.model_name, r.ai_type, r.hashtag, r.accuracy_score, r.created_at 