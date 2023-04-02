import test_twitter_domain as twt_test

from KZ_project.ml_pipeline.services.twitter_service.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.ml_pipeline.data_generator.data_manipulation import DataManipulation
from KZ_project.ml_pipeline.ai_model_creator.xgboost_forecaster import XgboostForecaster


import config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

INTERVAL = '1h'

def test_construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger() -> tuple:
    client_twt, tsa = twt_test.test_construct_twittercollection_and_tsentimentanalyser('en')
    daily, hourly = twt_test.test_get_sentiment_daily_hourly_scores('bnb', client_twt, tsa, hour=24*6)

    data = DataManipulation(config.BinanceConfig.SYMBOL, config.BinanceConfig.source, 
                            config.BinanceConfig.range_list, start_date='2023-01-18',
                            interval=INTERVAL, scale=config.BinanceConfig.SCALE, 
                            prefix_path='.', saved_to_csv=False,
                            logger=config.BinanceConfig.logger, client=config.BinanceConfig.client)
    return client_twt, tsa, daily, hourly, data

def test_get_tweet_sentiment_hourly(sent_tweets_hourly: pd.DataFrame()):
    sent_tweets = sent_tweets_hourly
    sent_tweets.Datetime = pd.to_datetime(sent_tweets.index)
    sent_tweets = sent_tweets.reset_index()
    sent_tweets.set_index('Datetime', inplace=True, drop=True)
    #sent_tweets.index = sent_tweets.index.tz_convert(None) # only hourly
    sent_tweets.index = sent_tweets.index + timedelta(hours=3)
    return sent_tweets


def test_composite_tweet_sentiment_and_data_manipulation(data: DataManipulation,
                                                         sent_tweets: pd.DataFrame()):

    print(data.df)
    df_price_ext = data.extract_features()
    df_price_ext.index = df_price_ext.index + timedelta(hours=3)
    df_final = tsa.concat_ohlc_compound_score(df_price_ext, sent_tweets)
    del df_price_ext
    #df_final = df_final.loc['2021-01-01':,:].copy()
    df_final = df_final.rename(columns={"compound_total":"twitter_sent_score"})
    df_final.dropna(inplace=True)
    return df_final

def test_backtest_prediction(X_pd, y_pred):
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
    
def test_trade_fee_net_returns(X_pd: pd.DataFrame()):
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

def test_predict_last_day_and_next_hour():
    xgb = XgboostForecaster(objective='binary', n_estimators=500, eta=0.01, max_depth=7, 
                    tree_method='gpu_hist', eval_metric='logloss')

    xgb.load_model('./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/btc/BTCUSDT_binance_model_price_1h_feature_numbers_225.json')

    X = df_final.drop(columns=['feature_label'], axis=1)

    ypred_reg = xgb.model.predict(X)
    
    test_backtest_prediction(X, ypred_reg)
    test_trade_fee_net_returns(X)
    pos = X["position"]
    print(f'X: {X}, position: {pos}')
    
    return ypred_reg[-1]

if __name__ == '__main__':
    client_twt, tsa, daily, hourly, data = test_construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger()
    hourly_tsa = test_get_tweet_sentiment_hourly(hourly)
    df_final = test_composite_tweet_sentiment_and_data_manipulation(data, hourly_tsa)
    next_candle_prediction = test_predict_last_day_and_next_hour()
    print(f'\nNext Candle Hourly Prediction for {config.BinanceConfig.SYMBOL} is: {next_candle_prediction}')
    
    

