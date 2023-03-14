from KZ_project.twitter.twitter_collection import TwitterCollection
from KZ_project.twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer

query_list = ['btc','eth','xrp','bnb','shib', 'ada', 'doge']

def test_construct_twittercollection_and_tsentimentanalyser(lang: str='en'):
    client_twitter = TwitterCollection()
    tsa = TweetSentimentAnalyzer(lang=lang)
    return client_twitter, tsa


def test_get_sentiment_daily_hourly_scores(symbol: str, 
                                           twitter_client: TwitterCollection,
                                           tsa: TweetSentimentAnalyzer):
    df_tweets = twitter_client.get_tweets_with_interval(symbol, 'en', hour=24*1, interval=1)
    print(f'######## Shape of {symbol} tweets df: {df_tweets.shape}')
    path_df = f'./data/tweets_data/{symbol}/'
    file_df = f'{symbol}_tweets.csv'
    daily_sents, hourly_sents = tsa.create_sent_results_df(symbol, df_tweets, path_df, saved=False)
    return daily_sents, hourly_sents
    


if __name__ == '__main__':
    
    ct, tsa = test_construct_twittercollection_and_tsentimentanalyser('en')
    daily, hourly = test_get_sentiment_daily_hourly_scores('bnb', ct, tsa)
    print(hourly)

    