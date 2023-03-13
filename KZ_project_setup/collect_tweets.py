from twitter.twitter_collection import TwitterCollection
from twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer

if __name__ == '__main__':

    client_twitter = TwitterCollection()
    tsa = TweetSentimentAnalyzer(lang='en')

    query_list = ['btc','eth','xrp','bnb']
    query_list_bist = ['xu100', 'bist', 'sasa','agyo', 'sngyo', 'sumas', 'orma']
    new_query = ['shib', 'ada', 'doge']
    deprem = ['hatay', 'antep', 'maras', 'adiyaman']
    
    
    for i in new_query[1:]:
        symbol = i
        print(f'\n############## SYMBOL - {i}')
        df_tweets = client_twitter.get_tweets_with_interval(symbol, 'en', hour=24*7, interval=2)
        print(f'######## Shape of {i} tweets df: {df_tweets.shape}')
        path_df = f'./KZ_project_setup/data/tweets_data/{symbol}/'
        file_df = f'{symbol}_tweets.csv'
        #df_tweets = client_twitter.get_tweets_df(symbol, path_df, file_df)
        #client_twitter.write_tweets_csv(df_tweets, path_df, file_df)
        daily_sents, hourly_sents = tsa.create_sent_results_df(symbol, df_tweets, path_df)
        

  