from datetime import datetime

class Tracker:
    def __init__(self, symbol: str, datetime_t: str, position: int, created_at=datetime.now()):
        self.symbol = symbol
        self.datetime_t = datetime_t
        self.position = position
        self.created_at = created_at
        
        
    def __eq__(self, other):
        if not isinstance(other, Tracker):
            return False
        cond1 = other.symbol == self.symbol
        cond2 = other.datetime_t == self.datetime_t
        cond3 = other.position == self.position
        return cond1 and cond2 and cond3
    
    def __hash__(self):
        return hash(self.symbol)
        
        
    def __repr__(self):
        return f"<Tracker {self.symbol}, datetime: {self.datetime_t}, position: {self.position}, created_at: {self.created_at}>"