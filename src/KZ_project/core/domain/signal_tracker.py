from datetime import datetime

class SignalTracker():
    def __init__(self, signal: str, datetime_t: str, position: int, forecast_model, created_at=datetime.now()):
        self.signal = signal
        self.datetime_t = datetime_t
        self.position = position
        self.created_at = created_at
        self.forecast_model = forecast_model