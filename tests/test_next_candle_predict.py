import test_twitter_domain as twt_test

from KZ_project.twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.data_pipelines.data_manipulation import DataManipulation
from KZ_project.dl_models.xgboost_forecaster import XgboostForecaster
from KZ_project.logger.logger import Logger

import config
import pandas as pd


INTERVAL = '1h'

def test_construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger() -> tuple:
    client_twt, tsa = twt_test.test_construct_twittercollection_and_tsentimentanalyser('en')
    daily, hourly = twt_test.test_get_sentiment_daily_hourly_scores('btc', client_twt, tsa)

    logger = Logger(config.BinanceConfig.LOG_PATH, config.BinanceConfig.LOG_FILE_NAME_PREFIX)
    data = DataManipulation(config.BinanceConfig.SYMBOL, config.BinanceConfig.source, 
                            config.BinanceConfig.range_list, start_date='2023-02-18',
                            interval=INTERVAL, scale=config.BinanceConfig.SCALE, 
                            prefix_path='.', saved_to_csv=False,
                            logger=logger, client=config.BinanceConfig.client)
    return client_twt, tsa, daily, hourly, data

def test_get_tweet_sentiment_hourly(sent_tweets_hourly: pd.DataFrame()):
    sent_tweets = sent_tweets_hourly
    sent_tweets.Datetime = pd.to_datetime(sent_tweets.index)
    sent_tweets = sent_tweets.reset_index()
    sent_tweets.set_index('Datetime', inplace=True, drop=True)
    #sent_tweets.index = sent_tweets.index.tz_convert(None) # only hourly
    return sent_tweets


def test_composite_tweet_sentiment_and_data_manipulation(data: DataManipulation,
                                                         sent_tweets: pd.DataFrame()):

    print(data.df)
    df_price_ext = data.extract_features()
    df_final = tsa.concat_ohlc_compound_score(df_price_ext, sent_tweets)
    del df_price_ext
    #df_final = df_final.loc['2021-01-01':,:].copy()
    df_final = df_final.rename(columns={"compound_total":"twitter_sent_score"})
    df_final.dropna(inplace=True)
    return df_final


def test_predict_last_day_and_next_hour():
    xgb = XgboostForecaster(objective='binary', n_estimators=500, eta=0.01, max_depth=7, 
                    tree_method='gpu_hist', eval_metric='logloss')

    xgb.load_model('./src/KZ_project/dl_models/model_stack/BTCUSDT_binance_model_price_1h_feature_numbers_225.json')

    X = df_final.drop(columns=['feature_label'], axis=1)

    ypred_reg = xgb.model.predict(X)
    print(f'prediction for 5 hours: {ypred_reg}')
    return ypred_reg[-1]

if __name__ == '__main__':
    client_twt, tsa, daily, hourly, data = test_construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger()
    hourly_tsa = test_get_tweet_sentiment_hourly(hourly)
    df_final = test_composite_tweet_sentiment_and_data_manipulation(data, hourly_tsa)
    next_candle_prediction = test_predict_last_day_and_next_hour()
    print(f'\nNext Candle Hourly Prediction for {config.BinanceConfig.SYMBOL} is: {next_candle_prediction}')
    
    

