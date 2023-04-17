from datetime import datetime

class ForecastModel():

    def __init__(self, symbol, source, feature_counts, model_name, interval, ai_type, hashtag=None, accuracy_score=None, created_at=datetime.now()):
        self.symbol = symbol
        self.source = source
        self.feature_counts = feature_counts
        self.model_name = model_name
        self.interval = interval
        self.ai_type = ai_type
        self.hashtag = hashtag
        self.accuracy_score = accuracy_score
        self.created_at = created_at
        self._crypto = None
        self._signals_tracker = []
        
    def add_signal_tracker(self, signals_tracker):
        if signals_tracker not in self._signals_tracker:
            self._signals_tracker.append(signals_tracker)
            signals_tracker._forecast_model = self

    def remove_signal_tracker(self, signals_tracker):
        if signals_tracker in self._signals_tracker:
            self._signals_tracker.remove(signals_tracker)
            signals_tracker._forecast_model = None

    def __repr__(self):
        return f"<ForecastModel {self.model_name}, symbol: {self.symbol}, interval: {self.interval}>"