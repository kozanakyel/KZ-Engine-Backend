import pandas as pd
from binance.client import Client
import pandas as ta
import numpy as np
import matplotlib.pyplot as plt

from binance import ThreadedWebsocketManager
from datetime import timedelta

from KZ_project.logger.logger import Logger
from KZ_project.binance_domain.binance_client import BinanceClient
from KZ_project.twitter.twitter_collection import TwitterCollection
from KZ_project.twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.data_pipelines.data_manipulation import DataManipulation
from KZ_project.dl_models.xgboost_forecaster import XgboostForecaster


import KZ_project.config as config


class AITrader():
    
    def __init__(self, symbol: str, name: str, bar_length, client: BinanceClient, logger: Logger=None):
        self.symbol = symbol
        self.name = name
        self.bar_length = bar_length
        self.data = pd.DataFrame(columns = ["Open", "High", "Low", "Close", "Volume", "Complete"])
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]   
        self.client = client
        self.postion = 0
        self.logger = logger
      
    def start_trading(self):
        
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length)
        # "else" to be added later in the course 
    
    
    def stream_candles(self, msg):
        
        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit = "ms") +  timedelta(hours = 3)
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms") +  timedelta(hours = 3)
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]
        
        start_date = (event_time - timedelta(days=10)).strftime('%Y-%m-%d')
    
        # print out
        print("Time: {} | Price: {} | Complete: {}".format(event_time, close, complete))
        
        if complete:
            #print(f'Write logic for prediction in this area')
            #print(f'start date: {start_date} and type {type(start_date)}')
            self.main_prediction(start_date=start_date, name=self.name, symbol=self.symbol)
            #self.execute_trades()
        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low, close, volume, complete]
        
    def stop_trading(self):
        self.twm.stop()
        
    def execute_trades(self): 
        if self.prepared_data["position"].iloc[-1] == 1: # if position is long -> go/stay long
            if self.position == 0:
                order = self.client.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG")  # NEW
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0: # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = self.client.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL")  # NEW
            self.position = 0
    
    def report_trade(self, order, going):  # NEW
        
        # extract data from order object
        side = order["side"]
        time = pd.to_datetime(order["transactTime"], unit = "ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)
        
        # calculate trading profits
        self.trades += 1
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units) 
        
        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]), 3) 
            cum_profits = round(np.sum(self.trade_values), 3)
        else: 
            real_profit = 0
            cum_profits = round(np.sum(self.trade_values[:-1]), 3)
        
        # print trade report
        print(2 * "\n" + 100* "-")
        print("{} | {}".format(time, going)) 
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(time, base_units, quote_units, price))
        print("{} | Profit = {} | CumProfits = {} ".format(time, real_profit, cum_profits))
        print(100 * "-" + "\n")
        
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
            
    ################# ######################################################
    
    def construct_twittercollection_and_tsentimentanalyser(self, lang: str='en'):
        client_twitter = TwitterCollection()
        tsa = TweetSentimentAnalyzer(lang=lang)
        return client_twitter, tsa
    
    def get_sentiment_daily_hourly_scores(self, name: str, 
                                           twitter_client: TwitterCollection,
                                           tsa: TweetSentimentAnalyzer,
                                           hour: int=24*1):
        #print(f'attribuites get interval: {name} {hour} ')
        df_tweets = twitter_client.get_tweets_with_interval(name, 'en', hour=hour, interval=1)
        #print(f'######## Shape of {name} tweets df: {df_tweets.shape}')
        #print(f'pure tweets hourly: {df_tweets.iloc[-1]}')
        path_df = f'./data/tweets_data/{name}/'
        file_df = f'{name}_tweets.csv'
        daily_sents, hourly_sents = tsa.create_sent_results_df(name, df_tweets, path_df, saved=False)
        #print(f'tweets hourly: {hourly_sents.iloc[-1]}')
        return daily_sents, hourly_sents
    

    def construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger(self, start_date, name: str, symbol) -> tuple:
        INTERVAL = '1h'
        client_twt, tsa = self.construct_twittercollection_and_tsentimentanalyser('en')
        daily, hourly = self.get_sentiment_daily_hourly_scores(name, client_twt, tsa)

        data = DataManipulation(symbol, config.BinanceConfig.source, 
                            config.BinanceConfig.range_list, start_date=start_date,
                            interval=INTERVAL, scale=config.BinanceConfig.SCALE, 
                            prefix_path='.', saved_to_csv=False, client=self.client)
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
        if self.name == 'btc':
            xgb.load_model('./src/KZ_project/dl_models/model_stack/btc/test_BTCUSDT_binance_model_price_1h_feature_numbers_225.json')
        elif self.name == 'bnb':
            xgb.load_model('./src/KZ_project/dl_models/model_stack/bnb/test_BNBUSDT_binance_model_price_1h_feature_numbers_225.json')

        X = df_final.drop(columns=['feature_label'], axis=1)
        self.prepared_data = X

        ypred_reg = xgb.model.predict(X)
        self.backtest_prediction(X, ypred_reg)
        self.trade_fee_net_returns(X)
    
        return X, ypred_reg[-1]
    
    def main_prediction(self, start_date, name, symbol):
        client_twt, tsa, daily, hourly, data = self.construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger(start_date, name, symbol)
        hourly_tsa = self.get_tweet_sentiment_hourly(hourly)
        df_final = self.composite_tweet_sentiment_and_data_manipulation(data, hourly_tsa, tsa)
        Xt, next_candle_prediction = self.predict_last_day_and_next_hour(df_final)
        print(f'\n\nlast row: {Xt}\nNext Candle Hourly Prediction for {self.symbol} is: {next_candle_prediction}\n\n')
    
    

    
    