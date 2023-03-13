import pandas as pd
from ..KZ_project.twitter.twitter_collection import TwitterCollection
from ..KZ_project.twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from ..KZ_project.data_pipelines.data_manipulation import DataManipulation
from ..KZ_project.dl_models.xgboost_forecaster import XgboostForecaster
from ..KZ_project.logger.logger import Logger

from sklearn.metrics import accuracy_score, confusion_matrix
import os


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)

SYMBOL = 'BTC-USD'
SYMBOL_NAME = 'Bitcoin'
SYMBOL_CUT = 'btc'
scale = 1
range_list = [i for i in range(5,21)]
range_list = [i*scale for i in range_list]
interval = '1h'
start_date = '2020-06-30'
end_date = '2022-07-01'
source = 'yahoo'
LOG_PATH = '../KZ_project_domain/logger' + os.sep + "logs"
LOG_FILE_NAME_PREFIX = f"log_{SYMBOL_CUT}_{start_date}_{end_date}"

logger = Logger(LOG_PATH, LOG_FILE_NAME_PREFIX)
tsa = TweetSentimentAnalyzer()
data = DataManipulation(SYMBOL, source, range_list, start_date=start_date, 
                        end_date=end_date, interval=interval, scale=scale, 
                        prefix_path='..', saved_to_csv=True,
                        logger=logger)
df_price = data.df.copy()

sent_tweets = pd.read_csv('../KZ_project_setup/notebooks/btc_archieve/btc_hourly_sent_score.csv')
#sent_tweets = pd.read_csv('../data/tweets_data/btc/btc_hour.csv')
sent_tweets.Datetime = pd.to_datetime(sent_tweets.Datetime)
sent_tweets.set_index('Datetime', inplace=True, drop=True)

sent_tweets.index = sent_tweets.index.tz_convert(None)

df_price_ext = data.extract_features()
close_col = df_price['close']

df_final = tsa.concat_ohlc_compound_score(df_price_ext, sent_tweets)
#df_final = df_final.loc['2021-01-01':,:].copy()
df_final = df_final.rename(columns={"compound_total":"twitter_sent_score"})
df_final.dropna(inplace=True)

y = df_final.feature_label
X = df_final.drop(columns=['feature_label'], axis=1)

eval_metric = 'logloss'
eval_metric = None
xgb = XgboostForecaster(objective='binary', n_estimators=500, eta=0.01, max_depth=7, 
                    tree_method='gpu_hist', eval_metric='logloss')
xgb.create_train_test_data(X, y, test_size=0.2)
xgb.fit()
xgb.save_model(f'../KZ_project_domain/dl_models/model_stack/model_price_{interval}_feature_numbers_{X.shape[1]}.json')
score = xgb.get_score()

if __name__ == '__main__':
    print(f'first score: {score}')
    #xgb.plot_learning_curves()
    xgb.get_model_names('../KZ_project_domain/dl_models/model_stack/')
    #best_params = xgb.bestparams_gridcv([100, 200], [0.1], [1, 3], verbose=3)

    ytest = xgb.y_test
    ypred_reg = xgb.model.predict(xgb.X_test)
    print(f'Last accuracy: {accuracy_score(ytest, ypred_reg)}')
    print(f'Confusion Matrix: {confusion_matrix(ytest, ypred_reg)}')


    n_feat = xgb.get_n_importance_features(10)
    print(n_feat)


