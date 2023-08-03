from datetime import timedelta

from KZ_project.ml_pipeline.ai_model_creator.engines.model_engine import ModelEngine
from KZ_project.Infrastructure.file_processor.data_checker import DataChecker
from KZ_project.ml_pipeline.ai_model_creator.forecasters.xgboost_binary_forecaster import XgboostBinaryForecaster
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
from KZ_project.ml_pipeline.data_pipeline.sentiment_feature_matrix_pipeline import SentimentFeaturedMatrixPipeline
from KZ_project.webapi.services import services
from KZ_project.webapi.entrypoints.flask_app import get_session

class ForecastEngine:
    
    def __init__(
        self, 
        data_creator: DataCreator, 
        hashtag: str, 
        data_checker: DataChecker=None, 
        is_backtest: bool=False, 
        is_twitter: bool=True
    ):
        self.data_creator = data_creator
        self.data_checker = data_checker
        self.hashtag = hashtag
        self.is_backtest = is_backtest
        self.is_twitter = is_twitter
        self.sentiment_featured_pipeline = SentimentFeaturedMatrixPipeline(
                                                    data_creator, 
                                                    data_checker, 
                                                    hashtag, 
                                                    is_twitter=self.is_twitter)
        
    def predict_next_candle(self, df_final):    
        forecaster = XgboostBinaryForecaster(eta=0.3)
        model_engine = ModelEngine(
            self.data_creator.symbol, 
            self.hashtag, 
            self.data_creator.source, 
            self.data_creator.interval,
            forecaster
        )
        dtt, y_pred, bt_json, acc_score = model_engine.create_model_and_strategy_return(df_final)
        self.ai_type = model_engine.ai_type
        
        if self.data_creator.interval[-1] == 'h':
            datetime_t = str(dtt + timedelta(hours=int(self.data_creator.interval[0])))
        elif self.data_creator.interval[-1] == 'd':
            datetime_t = str(dtt + timedelta(days=int(self.data_creator.interval[0])))
        
        return datetime_t, int(y_pred), bt_json
    
    
    def forecast_builder(self):
        sentiment_featured_matrix = self.sentiment_featured_pipeline.create_sentiment_aggregate_feature_matrix()
        datetime_t, next_candle_prediction, bt_json = self.predict_next_candle(sentiment_featured_matrix)
        last_candle_structure = self.data_creator.get_candlesticks(is_complete=False)
        print('last candlestick', last_candle_structure['candlestick_pattern'].iloc[-1])
        
        if not self.is_backtest:
            response_db = services.prediction_service_new_signaltracker(
                self.ai_type, 
                datetime_t, 
                next_candle_prediction,
                self.data_creator.symbol, 
                self.data_creator.interval, 
                self.hashtag, 
                self.sentiment_featured_pipeline.tweet_counts, 
                last_candle_structure['candlestick_pattern'].iloc[-1],
                bt_json, 
                get_session()
            )
            print(f'db commit signal: {response_db}')

        return self.ai_type, datetime_t, next_candle_prediction
    
    
    
        
        
        
    