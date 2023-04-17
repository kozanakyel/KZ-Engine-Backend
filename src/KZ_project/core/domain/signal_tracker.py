from datetime import datetime

class Signaltracker():
    def __init__(self, signal: str, datetime_t: str, position: int, created_at=datetime.now()):
        self.signal = signal
        self.datetime_t = datetime_t
        self.position = position
        self.created_at = created_at
        self._forecast_model = None