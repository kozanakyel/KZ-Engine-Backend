from datetime import timedelta
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score

from KZ_project.ml_pipeline.ai_model_creator.engines.model_engine import ModelEngine
from KZ_project.ml_pipeline.ai_model_creator.forecasters.xgboost_binary_forecaster import XgboostBinaryForecaster
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
from KZ_project.ml_pipeline.data_pipeline.sentiment_feature_matrix_pipeline import SentimentFeaturedMatrixPipeline
from KZ_project.core.interfaces.Iclient_service import IClientService

class Backtester():
    
    def __init__(
        self, 
        period: str, 
        client: IClientService, 
        data_creator: DataCreator
    ):
        self.client = client
        self.data_creator = data_creator
        self.period = period
        self.featured_matrix = self._create_featured_matrix()
        self.backtest_data = []
        
    def get_period(self) -> str:
        return self.period
    
    def set_period(self, period: str) -> None:
        self.period = period
        
    def _create_featured_matrix(self) -> pd.DataFrame:
        pipeline = SentimentFeaturedMatrixPipeline(self.data_creator, None, None, is_twitter=False)
        featured_matrix = pipeline.create_sentiment_aggregate_feature_matrix()
        
        return featured_matrix
        
    def _get_interval_df(self, start_index: int) -> pd.DataFrame:
        end_index = self._get_end_index(start_index)
        df = self.featured_matrix.iloc[start_index:end_index]
        return df
    
    def _get_end_index(self, start_index: int) -> int:
        ei = 0
        print(f' get interval type: {self.data_creator.interval}')
        if self.data_creator.interval == '1h':
            ei = start_index + self.period*24
        elif self.data_creator.interval == '1d':
            ei = start_index + self.period
        else:
            raise ValueError(f'Enbd index is not available end_index: {ei}')
        return ei
        
     
    def _predict_next_candle(self, df: pd.DataFrame) -> tuple:      
        model_engine = ModelEngine(self.data_creator.symbol, None, self.data_creator.source, self.data_creator.interval, is_backtest=True)
        # model_engine.xgb.load_model(f"./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/eth/test_ETHUSDT_binance_model_price_1h_feature_numbers_129.json")
        dtt, y_pred, bt_json, acc_score = model_engine.create_model_and_strategy_return(df)        
        
        return str(dtt + timedelta(hours=int(self.data_creator.interval[0]))), int(y_pred), bt_json, acc_score
    
    def _predict_next_candle_from_model(self, df: pd.DataFrame) -> tuple:      
        model_engine = ModelEngine(self.data_creator.symbol, None, self.data_creator.source, self.data_creator.interval, is_backtest=True)
        model_engine.xgb.load_model(f"./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/doge/test_DOGEUSDT_binance_model_price_1h_feature_numbers_129.json")
        # dtt, y_pred, bt_json, acc_score = model_engine.create_model_and_strategy_return(df)
        y = df.feature_label
        X = df.drop(columns=['feature_label'], axis=1)
        y_pred = model_engine.xgb.model.predict(X)
        acc_score = accuracy_score(y_pred, y)
        return acc_score
    
    def grid_backtest(self) -> tuple:
        fm = self.featured_matrix.copy()
        y = fm.feature_label
        x = fm.drop(columns=['feature_label'], axis=1)
        xgb = XgboostBinaryForecaster(early_stopping_rounds=0)
        xgb.create_train_test_data(x, y, test_size=0.2)
        model_gcv = xgb.model
        best_params = GridSearchableCV.bestparams_gridcv([100, 200, 400, 800, 1200], [0.1, 0.3, 0.5, 0.7, 0.9], [1, 3], model_gcv, xgb.X_train, xgb.y_train, verbose=3)
        return best_params    # 0.1, 1, 100
    
    
    def backtest(self, backtest_counts: int) -> float:
        fm = self.featured_matrix
        true_accuracy = []
        false_accuracy = []
        succes_predict = 0
        for i in range(backtest_counts):
            df = self._get_interval_df(i)
            # df.to_csv(f"./data/outputs/model_fine_tuned_data/prompt_{i}.csv")
            dt, signal, bt, accuracy_score = self._predict_next_candle(df)
            ei = self._get_end_index(i)
            actual_result = (fm.loc[fm.index[ei+1]]["log_return"] > 0).astype(int)
            # log_return = fm.loc[fm.index[ei+1]]["log_return"]
            # if accuracy_score > 0.50:
            self.backtest_data.append((dt, accuracy_score, signal, actual_result))
            
            if actual_result == signal:
                succes_predict = succes_predict + 1
                true_accuracy.append(accuracy_score)
            else:
                false_accuracy.append(accuracy_score)
        return succes_predict / len(self.backtest_data)
    

if __name__ == '__main__':
    from KZ_project.Infrastructure.services.binance_service.binance_client import BinanceClient
    from KZ_project.ml_pipeline.ai_model_creator.forecasters.gridsearchable_cv import GridSearchableCV
    from dotenv import load_dotenv
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret_key = os.getenv('BINANCE_SECRET_KEY')

    client = BinanceClient(api_key, api_secret_key) 
    data_creator = DataCreator(symbol="DOGEUSDT", source='binance', range_list=[i for i in range(5, 21)],
                                       period=None, interval="1h", start_date="2023-01-01", client=client)
    bt = Backtester(7, client, data_creator)
    score = bt._predict_next_candle_from_model(bt.featured_matrix)
    # result_score = bt.backtest(1)
    print(f'ACCURACY SKOR FOR LAST BACKTEST: {score} last shape: {bt.featured_matrix.shape}')
    
    # # Assuming self.backtest_data is a list of tuples
    # data = pd.DataFrame(bt.backtest_data, columns=['date', 'accuracy', 'signal', 'actual'])
    # data['date'] = pd.to_datetime(data['date'])
    
    
    # # Plot the data
    # plt.plot(data['date'], data['accuracy'], label='Accuracy')
    # # plt.plot(data['date'], data['signal'], label='Signal')
    # # plt.plot(data['date'], data['actual'], label='Actual')
    # plt.legend()
    # plt.show()
    
    # gcv = bt.grid_backtest()
    # print(f'best params: {gcv}')   # 0.5, 1, 400
    # best params: {'eta': 0.9, 'max_depth': 1, 'n_estimators': 1200}
    
        
    