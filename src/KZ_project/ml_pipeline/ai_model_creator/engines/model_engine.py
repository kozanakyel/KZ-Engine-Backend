from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import json

from KZ_project.ml_pipeline.ai_model_creator.engines.Ibacktestable import IBacktestable
from KZ_project.ml_pipeline.ai_model_creator.forecasters.gridsearchable_cv import GridSearchableCV
from KZ_project.ml_pipeline.ai_model_creator.forecasters.xgboost_binary_forecaster import XgboostBinaryForecaster
from KZ_project.ml_pipeline.data_generator.data_manipulation import DataManipulation
from KZ_project.ml_pipeline.services.twitter_service.tweet_sentiment_analyzer import TweetSentimentAnalyzer

from KZ_project.webapi.services import services
from KZ_project.webapi.entrypoints.flask_app import get_session
from KZ_project.Infrastructure import config

class ModelEngine(IBacktestable):
    
    def __init__(self, symbol, symbol_cut, source, interval):
        self.symbol = symbol
        self.source = source
        self.interval = interval
        self.symbol_cut = symbol_cut
        self.data_plot_path = f'./data/plots/model_evaluation/'
        self.model_plot_path = self.data_plot_path + f'{self.symbol_cut}/{self.symbol}_{self.source}_{self.interval}_model_backtest.png'
        self.model_importance_feature = self.data_plot_path + f'{self.symbol_cut}/{self.symbol}_{self.source}_{self.interval}_model_importance.png'
    
    def get_tweet_sentiment_hourly_filed(self, tweet_file):
        #sent_tweets = pd.read_csv('./data/archieve_data/btc_archieve/btc_hourly_sent_score.csv')
        sent_tweets = pd.read_csv(tweet_file)
        #sent_tweets = pd.read_csv('../data/tweets_data/btc/btc_hour.csv')  
        sent_tweets.Datetime = pd.to_datetime(sent_tweets.Datetime)
        
        if self.interval[-1] == 'd':
            #print('yes daily model')
            sent_tweets['Datetime'] = sent_tweets['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
            sent_tweets.Datetime = pd.to_datetime(sent_tweets.Datetime)
        sent_tweets.set_index('Datetime', inplace=True, drop=True)
        

        ss = datetime(2023, 1, 9, 0, 0, 0)
        sent_tweets = sent_tweets.loc[:ss]
        return sent_tweets
    
    
    def composite_tweet_sentiment_and_data_manipulation(self, data: DataManipulation,
                                                             tsa: TweetSentimentAnalyzer,
                                                         sent_tweets: pd.DataFrame()):

        df_price_ext = data.extract_features()
        
        df_final = tsa.concat_ohlc_compound_score(df_price_ext, sent_tweets)
        del df_price_ext
        df_final = df_final.rename(columns={"compound_total":"twitter_sent_score"})
        df_final.dropna(inplace=True)
        return df_final
    
    def create_retuns_data(self, X_pd, y_pred):
        X_pd["position"] = [y_pred[i] for i, _ in enumerate(X_pd.index)]    
        X_pd["strategy"] = X_pd.position.shift(1) * X_pd["log_return"]
        X_pd[["log_return", "strategy"]].sum().apply(np.exp)
        X_pd["cstrategy"] = X_pd["strategy"].cumsum().apply(np.exp) 
        X_pd["creturns"] = X_pd.log_return.cumsum().apply(np.exp) 
        
    def trade_fee_net_returns(self, X_pd: pd.DataFrame()):    
        X_pd["trades"] = X_pd.position.diff().fillna(0).abs()    
        commissions = 0.00075 # reduced Binance commission 0.075%
        other = 0.0001 # proportional costs for bid-ask spread & slippage (more detailed analysis required!)
        ptc = np.log(1 - commissions) + np.log(1 - other)
    
        X_pd["strategy_net"] = X_pd.strategy + X_pd.trades * ptc # strategy returns net of costs
        X_pd["cstrategy_net"] = X_pd.strategy_net.cumsum().apply(np.exp)
    
        X_pd[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8),  title = f"{self.symbol} - Buy and Hold")
        plt.savefig(self.model_plot_path)
        return X_pd[["creturns", "cstrategy", "cstrategy_net"]].to_json()
        
    def get_accuracy_score_for_xgboost_fit_separate_dataset(self, df_final: pd.DataFrame()):
        y = df_final.feature_label
        X = df_final.drop(columns=['feature_label'], axis=1)
        
        #print(f'shapes {X.shape, y.shape}')
    
        #X["twitter_sent_score"] = X["twitter_sent_score"].shift(1)
        #X["twitter_sent_score"][X.index[0]] = 0

        xgb = XgboostBinaryForecaster(n_estimators=500, eta=0.01, max_depth=7, 
                    tree_method='gpu_hist', eval_metric='logloss')
        self.ai_type = xgb.__class__.__name__
        xgb.create_train_test_data(X, y, test_size=0.2)
        
        xgb.fit()
    
        self.model_name = f'test_{self.symbol}_{self.source}_model_price_{self.interval}_feature_numbers_{X.shape[1]}.json'
        
        # try for instant model evaluation for one week    
        xgb.save_model(f'./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/{self.symbol_cut}/{self.model_name}')
        score = xgb.get_score()

        print(f'First score: {score}')
        #xgb.plot_learning_curves()
        # best_params = GridSearchableCV.bestparams_gridcv([100, 200], [0.1], [1, 3], verbose=3)

        # modelengine works
        ytest = xgb.y_test
        ypred_reg = xgb.model.predict(xgb.X_test)
        print(f'Last accuracy: {accuracy_score(ytest, ypred_reg)}')
        acc_score = accuracy_score(ytest, ypred_reg)
        print(f'Confusion Matrix:\n{confusion_matrix(ytest, ypred_reg)}')    
        
        res_str = services.save_crypto_forecast_model_service(acc_score, get_session(), self.symbol_cut, 
                                                          self.symbol, self.source, X.shape[1], self.model_name,
                                                          self.interval, self.ai_type)
        print(f'model engine model save: {res_str}')
    
        xgb.plot_feature_importance(
            self.model_importance_feature, 
            self.symbol
            )
    
        xtest = xgb.X_test
        self.create_retuns_data(xtest, ytest)
        bt_json = self.trade_fee_net_returns(xtest)
        print(f'x last row {xtest.index[-1]}\n prediction last candle {ytest[-1]}')
        print(f'bt: method inside: {json.dumps(bt_json)}')
        
        return xtest.index[-1], ytest[-1], json.dumps(bt_json)
        
    def start_model_engine(self, data:DataManipulation,
                            tsa: TweetSentimentAnalyzer, tweet_file):
        sent_tweets = self.get_tweet_sentiment_hourly_filed(tweet_file)
        df_final = self.composite_tweet_sentiment_and_data_manipulation(data, tsa, sent_tweets)
        #df_final = data.extract_features()
        datetime_t, prediction_t, bt_json_data = self.get_accuracy_score_for_xgboost_fit_separate_dataset(df_final)
        
if __name__ == '__main__':
    from KZ_project.Infrastructure import config
    asset_config = config.BinanceConfig()

    tsa = TweetSentimentAnalyzer()
    data = DataManipulation(asset_config.SYMBOL, asset_config.source, 
                        asset_config.range_list, start_date=asset_config.start_date, 
                        end_date=asset_config.end_date, interval=asset_config.interval_model, scale=asset_config.SCALE, 
                        prefix_path='.', saved_to_csv=False,
                        logger=asset_config.logger, client=asset_config.client)
    df_price = data.df.copy()
    
    model_engine = ModelEngine(asset_config.SYMBOL, asset_config.SYMBOL_CUT, asset_config.source, asset_config.interval_model)
    
    model_engine.start_model_engine(data, tsa, asset_config.tweet_file_hourly)
