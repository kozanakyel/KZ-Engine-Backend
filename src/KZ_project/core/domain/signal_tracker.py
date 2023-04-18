from datetime import datetime

class SignalTracker():
    def __init__(self, signal: str, ticker:str, 
                 tweet_counts:int, datetime_t: str, 
                 forecast_model, created_at=datetime.now()):
        self.signal = signal
        self.datetime_t = datetime_t
        self.tweet_counts = tweet_counts
        self.ticker = ticker
        self.created_at = created_at
        self.forecast_model = forecast_model
        
    def __repr__(self):
        return f"<SignalTracker {self.ticker}, datetime: {self.datetime_t}, position: {self.signal}, created_at: {self.created_at}>"