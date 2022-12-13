from twitter.twitter_collection import TwitterCollection
from twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer
import os

if __name__ == '__main__':

    client = TwitterCollection()

    query_list = ['btc','eth','xrp','bnb']
    query_list_bist = ['sumas', 'orma', 'xu100', 'bist', 'sasa']
    test_query = ['agyo', 'sngyo']
    new_query = ['doge', 'shib', 'ada']

    tsa = TweetSentimentAnalyzer()

    """
    for i in query_list:
        path_df = f'./KZ_project_setup/data/tweets_data/{i}/'
        file_df = f'{i}_tweets.csv'
        df = client.get_tweets_df(i, pathdf=path_df, filedf=file_df)
        df = tsa.cleaning_tweet_data(df)
        df = tsa.preprocessing_tweet_datetime(df)
        df = tsa.get_sentiment_scores(df)
        tsa.add_datetime_to_col(df)
        df_result_day = tsa.get_sent_with_mean_interval(df, interval='1d')
        df_result_day.to_csv(os.path.join(path_df, f'{i}_day.csv'))

        df_result_hour = tsa.get_sent_with_mean_interval(df, interval='1h')
        df_result_hour.to_csv(os.path.join(path_df, f'{i}_hour.csv'))
    
    """
    
  
    for i in query_list_bist:
        df_tweets = client.get_tweets_with_interval(i, 'tr', hour=24*7, interval=4)
        print(f'shape of {i} tweets df: {df_tweets.shape}')
        path_df = f'./KZ_project_setup/data/tweets_data/{i}/'
        file_df = f'{i}_tweets.csv'
        client.write_tweets_csv(df_tweets, path_df, file_df)

  