from datetime import timedelta
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
import requests
from KZ_project.Infrastructure import config
from KZ_project.core.adapters.forecastmodel_repository import ForecastModelRepository
from KZ_project.core.adapters.signaltracker_repository import SignalTrackerRepository
from KZ_project.ml_pipeline.ai_model_creator.xgboost_forecaster import XgboostForecaster
from KZ_project.ml_pipeline.data_generator.data_manipulation import DataManipulation
from KZ_project.ml_pipeline.services.twitter_service.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.ml_pipeline.services.twitter_service.twitter_collection import TwitterCollection

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from KZ_project.webapi.services import services

#orm.start_mappers()
engine = create_engine(config.get_postgres_uri())
get_session = sessionmaker(bind=engine)
#orm.metadata.create_all(engine)


class ForecastEngineHourly():
    
    def __init__(self, hastag, symbol, lang: str="en"):
        self.client_twitter = TwitterCollection()
        self.tsa = TweetSentimentAnalyzer(lang=lang)
        self.hastag = hastag
        self.symbol = symbol
     
    def get_sentiment_daily_hourly_scores(self, hastag: str, 
                                           twitter_client: TwitterCollection,
                                           tsa: TweetSentimentAnalyzer,
                                           hour: int=24*1):
        #print(f'attribuites get interval: {hastag} {hour} ')
        df_tweets = twitter_client.get_tweets_with_interval(hastag, 'en', hour=hour, interval=2)
        print(f'######## Shape of {hastag} tweets df: {df_tweets.shape}')
        self.tweet_counts = df_tweets.shape[0]
        #print(f'pure tweets hourly: {df_tweets.iloc[-1]}')
        path_df = f'./data/tweets_data/{hastag}/'
        file_df = f'{hastag}_tweets.csv'
        daily_sents, hourly_sents = tsa.create_sent_results_df(hastag, df_tweets, path_df, saved=False)
        #print(f'tweets hourly: {hourly_sents.iloc[-1]}')
        return daily_sents, hourly_sents   
    
    def construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger(self, start_date, hastag: str, symbol) -> tuple:
        INTERVAL = '1h'
        client_twt, tsa = self.client_twitter, self.tsa
        daily, hourly = self.get_sentiment_daily_hourly_scores(hastag, client_twt, tsa)

        data = DataManipulation(symbol, config.BinanceConfig.source, 
                            config.BinanceConfig.range_list, start_date=start_date,
                            interval=config.BinanceConfig.interval, scale=config.BinanceConfig.SCALE, 
                            prefix_path='.', saved_to_csv=False, client=config.BinanceConfig.client)
        #print(f'df shape: {data.df.iloc[-1]}')
        return client_twt, tsa, daily, hourly, data
    
    def get_tweet_sentiment_hourly(self, sent_tweets_hourly: pd.DataFrame()):
        sent_tweets = sent_tweets_hourly
        sent_tweets.Datetime = pd.to_datetime(sent_tweets.index)
        sent_tweets = sent_tweets.reset_index()
        sent_tweets.set_index('Datetime', inplace=True, drop=True)
        #sent_tweets.index = sent_tweets.index.tz_convert(None) # only hourly
        sent_tweets.index = sent_tweets.index + timedelta(hours=4)
        return sent_tweets
    
    def composite_tweet_sentiment_and_data_manipulation(self, data: DataManipulation,
                                                         sent_tweets: pd.DataFrame(),
                                                         tsa: TweetSentimentAnalyzer):

        df_price_ext = data.extract_features()
        df_price_ext.index = df_price_ext.index + timedelta(hours=3)
        #print(f'df extract features: {df_price_ext.iloc[-1]}')
        df_final = tsa.concat_ohlc_compound_score(df_price_ext, sent_tweets)
        #print(f'df tweets features: {df_final.iloc[-1]}')
        del df_price_ext
        #df_final = df_final.loc['2021-01-01':,:].copy()
        df_final = df_final.rename(columns={"compound_total":"twitter_sent_score"})
        df_final.dropna(inplace=True)
        return df_final 
    
    def backtest_prediction(self, X_pd, y_pred):
        X_pd["position"] = [y_pred[i] for i, _ in enumerate(X_pd.index)]
    
        print(X_pd[["log_return", "position"]]) 
    
        X_pd["strategy"] = X_pd.position.shift(1) * X_pd["log_return"]
        X_pd[["log_return", "strategy"]].sum().apply(np.exp)
        X_pd["cstrategy"] = X_pd["strategy"].cumsum().apply(np.exp) 
        X_pd["creturns"] = X_pd.log_return.cumsum().apply(np.exp) 

        X_pd.creturns.plot(figsize = (12, 8), title = "BTC/USDT - Buy and Hold", fontsize = 12)
        plt.show()

        X_pd[["creturns", "cstrategy"]].plot(figsize = (12 , 8), fontsize = 12)
        plt.show()
        
    def trade_fee_net_returns(self, X_pd: pd.DataFrame()):
        val_counts = X_pd.position.value_counts()
        print(f'val counts: {val_counts}')
    
        X_pd["trades"] = X_pd.position.diff().fillna(0).abs()
        trade_val_count = X_pd.trades.value_counts()
        print(f'trade val counts: {trade_val_count}')
    
        commissions = 0.00075 # reduced Binance commission 0.075%
        other = 0.0001 # proportional costs for bid-ask spread & slippage (more detailed analysis required!)
        ptc = np.log(1 - commissions) + np.log(1 - other)
    
        X_pd["strategy_net"] = X_pd.strategy + X_pd.trades * ptc # strategy returns net of costs
        X_pd["cstrategy_net"] = X_pd.strategy_net.cumsum().apply(np.exp)
    
        X_pd[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8))
        plt.show()
        
    def predict_last_day_and_next_hour(self, df_final):
        xgb = XgboostForecaster(objective='binary', n_estimators=500, eta=0.01, max_depth=7, 
                    tree_method='gpu_hist', eval_metric='logloss')
        self.ai_type = xgb.__class__.__name__
        fm_model = services.get_forecast_model(
                            self.symbol, 
                            config.BinanceConfig.interval,
                            self.ai_type)
        fm_model_name = fm_model.model_name
        xgb.load_model(f'./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/{self.hastag}/{fm_model_name}')
        print(f'./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/{self.hastag}/{fm_model_name}')
        """
        if self.hastag == 'btc':
            xgb.load_model('./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/btc/test_BTCUSDT_binance_model_price_2h_feature_numbers_225.json')
        elif self.hastag == 'bnb':
            xgb.load_model('./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/bnb/test_BNBUSDT_binance_model_price_2h_feature_numbers_225.json')
        """
        

        X = df_final.drop(columns=['feature_label'], axis=1)
        self.prepared_data = X

        ypred_reg = xgb.model.predict(X)
        self.backtest_prediction(X, ypred_reg)
        self.trade_fee_net_returns(X)
        
    
        return str(X.index[-1] + timedelta(hours=2)), int(ypred_reg[-1])
    
    def prediction_service(self, symbol, Xt, next_candle_prediction):
        url = config.get_api_url()
        r = requests.post(
            f"{url}/allocate_tracker", json={"symbol": symbol,
                                             "datetime_t": Xt,
                                             "position": next_candle_prediction
                                             }
        )
        print(f'Service tracker current status code is {r.status_code}')
        
    def prediction_service_new_signaltracker(self, ai_type, Xt, next_candle_prediction):
        session = get_session()
        repo = SignalTrackerRepository(session)
        repo_fm = ForecastModelRepository(session)
        try: 
            result_fm = services.get_forecast_model(
                self.symbol, 
                config.BinanceConfig.interval,
                ai_type,
                repo_fm,
                session
            )
            services.add_signal_tracker(
                next_candle_prediction,
                self.hastag,
                self.tweet_counts,
                Xt,
                result_fm,
                repo,
                session,
            )
        except (services.InvalidName) as e:
            return f'error occyred for signal tracker {self.symbol}'
        return f'Succes signaltracker for {self.symbol}'
    
    def forecast_builder(self, start_date):
        client_twt, tsa, daily, hourly, data = self.construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger(start_date, self.hastag, self.symbol)
        hourly_tsa = self.get_tweet_sentiment_hourly(hourly)
        df_final = self.composite_tweet_sentiment_and_data_manipulation(data, hourly_tsa, tsa)
        Xt, next_candle_prediction = self.predict_last_day_and_next_hour(df_final)
        print(f'\n\nlast row: {Xt}\nNext Candle Hourly Prediction for {self.symbol} is: {next_candle_prediction}\n\n')
        self.prediction_service(symbol=self.symbol, Xt=Xt, next_candle_prediction=next_candle_prediction)
        self.prediction_service_new_signaltracker(self.ai_type, Xt, next_candle_prediction)
        
        
        
    