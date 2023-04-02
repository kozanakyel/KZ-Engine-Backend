from datetime import datetime

class AIModel:
    def __init__(self, symbol: str, source: str, feature_counts: int,
                 model_name: str, ai_type: str, hashtag: str, accuracy_score: float, created_at=datetime.now()):
        self.symbol = symbol
        self.source = source
        self.feature_counts = feature_counts
        self.model_name = model_name
        self.ai_type = ai_type
        self.hashtag = hashtag
        self.accuracy_score = accuracy_score
        self.created_at = created_at
        
    @property    
    def get_filepath(self):
        return f"./src/KZ_project/dl_models/model_stack/{self.hashtag}/{self.model_name}"
               
    def __repr__(self):
        return f"<AIModel {self.symbol}, source: {self.source}, model_name: {self.model_name}>"