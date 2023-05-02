from datetime import timedelta

from KZ_project.ml_pipeline.ai_model_creator.engines.model_engine import ModelEngine
from KZ_project.ml_pipeline.data_generator.data_checker import DataChecker
from KZ_project.ml_pipeline.data_generator.data_creator import DataCreator
from KZ_project.ml_pipeline.data_generator.sentiment_feature_matrix_pipeline import SentimentFeaturedMatrixPipeline
from KZ_project.webapi.services import services
from KZ_project.webapi.entrypoints.flask_app import get_session

class ForecastEngine():
    
    def __init__(self, data_creator: DataCreator, hashtag, data_checker: DataChecker=None):
        self.data_creator = data_creator
        self.data_checker = data_checker
        self.hashtag = hashtag
        self.sentiment_featured_pipeline = SentimentFeaturedMatrixPipeline(data_creator, data_checker, hashtag)
        
    def predict_last_day_and_next_hour(self, df_final):      
        model_engine = ModelEngine(self.data_creator.symbol, self.hashtag, 'binance', self.data_creator.interval)
        dtt, y_pred, bt_json = model_engine.get_accuracy_score_for_xgboost_fit_separate_dataset(df_final)
        self.ai_type = model_engine.ai_type
        
        return str(dtt + timedelta(hours=int(self.data_creator.interval[0]))), int(y_pred)
    
    def forecast_builder(self):
        sentiment_featured_matrix = self.sentiment_featured_pipeline.create_sentiment_aggregate_feature_matrix()
        Xt, next_candle_prediction = self.predict_last_day_and_next_hour(sentiment_featured_matrix)
        response_db = services.prediction_service_new_signaltracker(self.ai_type, Xt, next_candle_prediction,
                                                  self.data_creator.symbol, self.data_creator.interval, self.hashtag, 
                                                  self.sentiment_featured_pipeline.tweet_counts, get_session())
        print(f'db commit signal: {response_db}')

        return self.ai_type, Xt, next_candle_prediction
        
        
        
    