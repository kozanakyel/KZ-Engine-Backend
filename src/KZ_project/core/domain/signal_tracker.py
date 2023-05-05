from datetime import datetime

class SignalTracker():
    def __init__(self, signal:int, ticker:str, 
                 tweet_counts:int, datetime_t: str, 
                 backtest_returns_data:str, forecast_model):
        self.signal = signal
        self.datetime_t = datetime_t
        self.tweet_counts = tweet_counts
        self.ticker = ticker
        self.backtest_returns_data = backtest_returns_data
        self.forecast_model = forecast_model
        
    def json(self):
        return {
            'signal':self.signal,
            'datetime_t':self.datetime_t,
            'tweet_counts':self.tweet_counts,
            'ticker':self.ticker,
            'backtest_returns_data':self.backtest_returns_data,
            'forecast_model':self.forecast_model.json()
            }  
        
    def __repr__(self):
        return f"<SignalTracker {self.ticker}, datetime: {self.datetime_t}, position: {self.signal}>"