from __future__ import annotations
from KZ_project.domain.asset import Asset, allocate_tracker
from KZ_project.domain.aimodel import AIModel
from KZ_project.adapters.repository import AbstractBaseRepository
from KZ_project.domain.tracker import Tracker
from KZ_project.domain.asset import InvalidSymbol



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