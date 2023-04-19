from datetime import datetime

class ForecastModel():

    def __init__(self, symbol, source, feature_counts, model_name, interval, ai_type, hashtag, accuracy_score, crypto, created_at=datetime.now()):
        self.symbol = symbol
        self.source = source
        self.feature_counts = feature_counts
        self.model_name = model_name
        self.interval = interval
        self.ai_type = ai_type
        self.hashtag = hashtag
        self.accuracy_score = accuracy_score
        self.created_at = created_at
        self.crypto = crypto
        
    def json(self):
        return {
           'symbol':self.symbol,
           'source':self.source,
           'feature_counts':self.feature_counts,
           'model_name':self.model_name,
           'interval':self.interval,
           'ai_type':self.ai_type,
           'hashtag':self.hashtag,
           'accuracy_score':self.accuracy_score,
           'created_at':self.created_at,
           'crypto':self.crypto.json() 
        }
        
    def __eq__(self, other):
        if not isinstance(other, ForecastModel):
            return False
        c1 = str(other) == str(self)
        c2 = str(self.created_at) == str(other.created_at)
        return c1 and c2

    def __repr__(self):
        return f"<ForecastModel {self.model_name}, symbol: {self.symbol}, interval: {self.interval}, aitype: {self.ai_type}>"