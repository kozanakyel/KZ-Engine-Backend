import pandas as pd
from twitter.twitter_collection import TwitterCollection
from twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)

for i in range(0, 10):
    print(f'loop: {i}')
    chunksize = 100000
    list_of_dataframes = []
    for df in pd.read_csv('./KZ_project_setup/notebooks/btc_archieve/bitcoin-tweets-2022.csv', chunksize=chunksize, lineterminator='\n'):
        list_of_dataframes.append(df)
    temp_tweets = pd.concat(list_of_dataframes)

    temp_tweets.reset_index(drop=True)
    l1 = i*int(len(temp_tweets)/10)
    l2 = (i+1)*int(len(temp_tweets)/10)

    temp1 = temp_tweets.iloc[l1:l2,:]

    del temp_tweets

    temp1 = temp1.drop(columns=['date'], axis=1)
    temp1.columns = ['created_at', 'username', 'text']
    temp1 = temp1[['created_at', 'text', 'username']]

    temp1.created_at = pd.to_datetime(temp1.created_at) 

    client_twitter = TwitterCollection()
    tsa = TweetSentimentAnalyzer()
#df = tsa.cleaning_tweet_data(temp1)
#df = tsa.preprocessing_tweet_datetime(temp1)
    df = tsa.get_sentiment_scores(temp1)
#tsa.add_datetime_to_col(df)

    df.to_csv(f'./KZ_project_setup/notebooks/btc_archieve/btc_sent_{i+10}.csv')

    del temp1
    del df

