from KZ_project.domain.tracker import Tracker
from typing import Optional, List, Set


class Asset:
    def __init__(self, symbol: str, source: str):
        self.symbol = symbol
        self.source = source
        self._allocations_tracker = set()
        
    def allocate_tracker(self, tracker: Tracker):
        if self.can_allocate(tracker):
            self._allocations_tracker.add(tracker)
            
    def deallocate(self, tracker: Tracker):
        if tracker in self._allocations_tracker:
            self._allocations_tracker.remove(tracker)
            
    def can_allocate(self, tracker: Tracker) -> bool:
        return tracker.symbol == self.symbol
    
    def __eq__(self, other):
        if not isinstance(other, Asset):
            return False
        cond1 = other.symbol == self.symbol
        cond2 = other.source == self.source
        return cond1 and cond2
    
    def __hash__(self):
        return hash(self.symbol)
        
    def __repr__(self):
        return f"<Asset {self.symbol}, source: {self.source}>"
    
    
    
class InvalidSymbol(Exception):
    pass


def allocate_tracker(tracker: Tracker, assets: List[Asset]) -> str:
    try:
        asset = next(a for a in assets if a.can_allocate(tracker))
        asset.allocate_tracker(tracker)
        return asset.symbol, tracker.position
    except StopIteration:
        raise InvalidSymbol(f"Symbol not included {tracker.symbol}")