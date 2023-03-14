import pandas as pd

from KZ_project.twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.data_pipelines.data_manipulation import DataManipulation
from KZ_project.dl_models.xgboost_forecaster import XgboostForecaster
from KZ_project.logger.logger import Logger

from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)

import config

logger = Logger(config.LOG_PATH, config.LOG_FILE_NAME_PREFIX)
tsa = TweetSentimentAnalyzer()
data = DataManipulation(config.SYMBOL, config.source, config.range_list, start_date=config.start_date, 
                        end_date=config.end_date, interval=config.interval, scale=config.SCALE, 
                        prefix_path='..', saved_to_csv=False,
                        logger=logger)
df_price = data.df.copy()

def test_get_tweet_sentiment_daily():
    sent_tweets = pd.read_csv('./data/archieve_data/btc_archieve/btc_daily_sent_score.csv')
    #sent_tweets = pd.read_csv('../data/tweets_data/btc/btc_hour.csv')  
    sent_tweets.Datetime = pd.to_datetime(sent_tweets.Datetime)
    sent_tweets.set_index('Datetime', inplace=True, drop=True)

    #sent_tweets.index = sent_tweets.index.tz_convert(None)   for only hourly
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


def test_get_accuracy_score_for_xgboost_fit_separate_dataset(df_final: pd.DataFrame()):
    y = df_final.feature_label
    X = df_final.drop(columns=['feature_label'], axis=1)

    eval_metric = 'logloss'
    eval_metric = None
    xgb = XgboostForecaster(objective='binary', n_estimators=500, eta=0.01, max_depth=7, 
                    tree_method='gpu_hist', eval_metric='logloss')
    xgb.create_train_test_data(X, y, test_size=0.2)
    xgb.fit()
    xgb.save_model(f'./src/KZ_project/dl_models/model_stack/{config.SYMBOL}_model_price_{config.interval}_feature_numbers_{X.shape[1]}.json')
    score = xgb.get_score()

    print(f'first score: {score}')
    #xgb.plot_learning_curves()
    xgb.get_model_names('./src/KZ_project/dl_models/model_stack/')
    #best_params = xgb.bestparams_gridcv([100, 200], [0.1], [1, 3], verbose=3)

    ytest = xgb.y_test
    ypred_reg = xgb.model.predict(xgb.X_test)
    print(f'Last accuracy: {accuracy_score(ytest, ypred_reg)}')
    print(f'Confusion Matrix: {confusion_matrix(ytest, ypred_reg)}')


    n_feat = xgb.get_n_importance_features(10)
    print(n_feat)
    
    xgb.plot_fature_importance()
    
def test_daily_model_train_save_tweet_composite_get_result_for_fetaure_importance():
    sent_tweets = test_get_tweet_sentiment_daily()
    df_final = test_composite_tweet_sentiment_and_data_manipulation(data, sent_tweets)
    test_get_accuracy_score_for_xgboost_fit_separate_dataset(df_final)


if __name__ == "__main__":
    test_daily_model_train_save_tweet_composite_get_result_for_fetaure_importance()

