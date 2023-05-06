from datetime import timedelta

from KZ_project.ml_pipeline.ai_model_creator.engines.model_engine import ModelEngine
from KZ_project.ml_pipeline.data_generator.data_creator import DataCreator
from KZ_project.ml_pipeline.data_generator.sentiment_feature_matrix_pipeline import SentimentFeaturedMatrixPipeline
from KZ_project.ml_pipeline.services.service_client.abstract_service_client import IServiceClient


class Backtester():
    
    def __init__(self, period: str, 
                 client: IServiceClient, data_creator: DataCreator):
        self.client = client
        self.data_creator = data_creator
        self.period = period
        self.featured_matrix = self.create_featured_matrix()
        self.backtest_data = []
        
    def create_featured_matrix(self):
        pipeline = SentimentFeaturedMatrixPipeline(self.data_creator, None, None, is_twitter=False)
        featured_matrix = pipeline.create_sentiment_aggregate_feature_matrix()
        return featured_matrix
        
    def get_interval_df(self, start_index):
        end_index = self.get_end_index(start_index)
        df = self.featured_matrix.iloc[start_index:end_index]
        return df
    
    def get_end_index(self, start_index):
        return start_index + self.period*24
     
    def predict_next_hour(self, df):      
        model_engine = ModelEngine(self.data_creator.symbol, None, self.data_creator.source, self.data_creator.interval, is_backtest=True)
        dtt, y_pred, bt_json, acc_score = model_engine.get_accuracy_score_for_xgboost_fit_separate_dataset(df)
        
        return str(dtt + timedelta(hours=int(self.data_creator.interval[0]))), int(y_pred), bt_json, acc_score
    
    def backtest(self):
        fm = backtester.create_featured_matrix()
        sum = 0
        for i in range(2400):
            df = backtester.get_interval_df(i)
            dt, signal, bt, ac = backtester.predict_next_hour(df)
            ei = backtester.get_end_index(i)
            actual_result = (fm.loc[fm.index[ei+1]]["log_return"] > 0).astype(int)
            self.backtest_data.append((dt, ac, signal, actual_result))
            if actual_result == signal:
                sum = sum + 1
        return sum / len(self.backtest_data)
    

if __name__ == '__main__':
    from KZ_project.ml_pipeline.services.binance_service.binance_client import BinanceClient
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret_key = os.getenv('BINANCE_SECRET_KEY')

    client = BinanceClient(api_key, api_secret_key) 
    data_creator = DataCreator(symbol="BNBUSDT", source='binance', range_list=[i for i in range(5,21)],
                                       period=None, interval="1h", start_date="2022-02-06", client=client)
    backtester = Backtester(7, client, data_creator)
    # print('created backtester')
    # fm = backtester.create_featured_matrix()
    # print('created fmmmm')
    # df = backtester.get_interval_df(0)
    # print('get df interval fmmmm')
    # dt, signal, bt, ac = backtester.predict_next_hour(df)
    # print('get prediction')
    # ei = backtester.get_end_index(0)
    # print(f'dt: {dt}, signal: {signal}, actual: {(fm.loc[fm.index[ei+1]]["log_return"] > 0).astype(int)}, acc_score: {ac}')
    
    result_score = backtester.backtest()
    print(f'ACCURACY SKOR FOR LAST BACKTEST: {result_score} list: {backtester.backtest_data}')
    
        
    