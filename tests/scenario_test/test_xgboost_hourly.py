import pandas as pd

from KZ_project.ml_pipeline.services.twitter_service.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.ml_pipeline.data_generator.data_manipulation import DataManipulation
from KZ_project.ml_pipeline.ai_model_creator.xgboost_forecaster import XgboostForecaster
from KZ_project.Infrastructure.logger.logger import Logger

from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)

import config
binance_config = config.BinanceConfig()

INTERVAL = '1h'

logger = Logger(binance_config.LOG_PATH, binance_config.LOG_FILE_NAME_PREFIX)
tsa = TweetSentimentAnalyzer()
data = DataManipulation(binance_config.SYMBOL, binance_config.source, 
                        binance_config.range_list, start_date=binance_config.start_date, 
                        end_date=binance_config.end_date, interval=INTERVAL, scale=binance_config.SCALE, 
                        prefix_path='.', saved_to_csv=False,
                        logger=logger, client=binance_config.client)
df_price = data.df.copy()

def test_get_tweet_sentiment_hourly():
    #sent_tweets = pd.read_csv('./data/archieve_data/btc_archieve/btc_hourly_sent_score.csv')
    sent_tweets = pd.read_csv('./data/tweets_data/btc/btc_hour.csv')
    #sent_tweets = pd.read_csv('../data/tweets_data/btc/btc_hour.csv')  
    sent_tweets.Datetime = pd.to_datetime(sent_tweets.Datetime)
    sent_tweets.set_index('Datetime', inplace=True, drop=True)

    #sent_tweets.index = sent_tweets.index.tz_convert(None) # only hourly
    return sent_tweets

def test_composite_tweet_sentiment_and_data_manipulation(data: DataManipulation,
                                                         sent_tweets: pd.DataFrame()):

    df_price_ext = data.extract_features()
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

    X_pd.creturns.plot(figsize = (12, 8), title = "BNB/USDT - Buy and Hold", fontsize = 12)
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


def test_get_accuracy_score_for_xgboost_fit_separate_dataset(df_final: pd.DataFrame()):
    y = df_final.feature_label
    X = df_final.drop(columns=['feature_label'], axis=1)
    
    X["twitter_sent_score"] = X["twitter_sent_score"].shift(1)
    X["twitter_sent_score"][X.index[0]] = 0

    eval_metric = 'logloss'
    eval_metric = None
    xgb = XgboostForecaster(objective='binary', n_estimators=500, eta=0.01, max_depth=7, 
                    tree_method='gpu_hist', eval_metric='logloss')
    xgb.create_train_test_data(X, y, test_size=0.2)
    xgb.fit()
    xgb.save_model(f'./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/btc/test_{binance_config.SYMBOL}_{binance_config.source}_model_price_{INTERVAL}_feature_numbers_{X.shape[1]}.json')
    score = xgb.get_score()

    print(f'first score: {score}')
    #xgb.plot_learning_curves()
    xgb.get_model_names('./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/')
    #best_params = xgb.bestparams_gridcv([100, 200], [0.1], [1, 3], verbose=3)

    ytest = xgb.y_test
    ypred_reg = xgb.model.predict(xgb.X_test)
    print(f'Last accuracy: {accuracy_score(ytest, ypred_reg)}')
    print(f'Confusion Matrix: {confusion_matrix(ytest, ypred_reg)}')


    n_feat = xgb.get_n_importance_features(10)
    print(n_feat)
    
    xgb.plot_fature_importance()
    
    ypr = xgb.model.predict(X)
    test_backtest_prediction(X, ypr)
    test_trade_fee_net_returns(X)
       

    
def test_daily_model_train_save_tweet_composite_get_result_for_fetaure_importance():
    sent_tweets = test_get_tweet_sentiment_hourly()
    df_final = test_composite_tweet_sentiment_and_data_manipulation(data, sent_tweets)
    #df_final = data.extract_features()
    test_get_accuracy_score_for_xgboost_fit_separate_dataset(df_final)


if __name__ == "__main__":
    test_daily_model_train_save_tweet_composite_get_result_for_fetaure_importance()