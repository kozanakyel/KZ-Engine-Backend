import re, tweepy, datetime, time, json, csv
from tweepy import OAuthHandler
import tweepy
import pandas as pd
import os
from dotenv import load_dotenv
import datetime
from datetime import datetime as dt 
from datetime import timedelta as td

load_dotenv()

access_token=os.getenv('TW_access_token')
access_token_secret=os.getenv('TW_access_token_secret')
consumer_key=os.getenv('TW_consumer_key')
consumer_secret=os.getenv('TW_consumer_secret')
MY_BEARER_TOKEN=os.getenv('TW_BEARER_TOKEN')

class TwitterCollection():
    def __init__(self, access_token=access_token, access_token_secret=access_token_secret,
                    consumer_key=consumer_key, consumer_secret=consumer_secret, bearer_token=MY_BEARER_TOKEN):
        self.access_token = access_token
        self.access_token_secret=access_token_secret
        self.consumer_key=consumer_key
        self.consumer_secret=consumer_secret
        self.bearer_token=bearer_token
        self.connect_twitter()
        self.client = tweepy.Client(bearer_token=self.bearer_token)

    def connect_twitter(self):
        try:
            auth = OAuthHandler(self.consumer_key, self.consumer_secret)
            auth.set_access_token(self.access_token, self.access_token_secret)
            api = tweepy.API(auth, wait_on_rate_limit=True)
            print("Authentication Successfull")
        except:
            print("Error: Authentication Failed")

    def get_tweets(self, search_query: str, lang: str, start_time: str, end_time: str):
        query = f"#{search_query} lang:{lang} -is:retweet"
        tweets = self.client.search_recent_tweets(query=query,
                                         start_time=start_time,
                                         end_time=end_time,
                                         tweet_fields = ["created_at", "text", "source"],
                                         user_fields = ["name", "username", "location", "verified", "description"],
                                         max_results = 100,
                                         expansions='author_id'
                                        ) 
        return tweets

    def converts_tweets_pd(self, tweets: dict) -> pd.DataFrame():
        tweet_info_ls = []
        for user in tweets.includes.get('users', ''):
            for tweet, user in zip(tweets.data, tweets.includes['users']):
                tweet_info = {
                'created_at': tweet.created_at,
                'text': tweet.text,
                'source': tweet.source,
                'name': user.name,
                'username': user.username,
                'location': user.location,
                'verified': user.verified,
                'description': user.description
                }
                tweet_info_ls.append(tweet_info)
        tweets_df = pd.DataFrame(tweet_info_ls)
        return tweets_df


    def get_tweets_with_interval(self, hashtag: str, lang: str, start_time=None, finish_time=None, hour=24, interval =1) -> pd.DataFrame():
        now = datetime.datetime.now(datetime.timezone.utc)
        if start_time == None:
            start_time = now - td(hours=hour)
        if finish_time == None:
            finish_time = now
        end_time = start_time + td(hours=interval)

        result_tweets = pd.DataFrame()
        while end_time <= finish_time - td(hours=1):
            temp_tweets = self.get_tweets(hashtag, lang, start_time.isoformat(), end_time.isoformat())
            df_temp_tweets = self.converts_tweets_pd(temp_tweets)
            result_tweets = pd.concat([df_temp_tweets, result_tweets], ignore_index=True)
            start_time = start_time + td(hours=1)
            end_time = end_time + td(hours=1)
        return result_tweets


    def write_tweets_csv(self, df: pd.DataFrame(), pathdf: str, filedf: str) -> None:
        if not os.path.exists(os.path.join(pathdf, filedf)):
            os.makedirs(pathdf, exist_ok=True)
            with open(os.path.join(pathdf, filedf), mode='a'): pass
            df['index_col'] = df.index 
            df = df.set_index('index_col')
            df.to_csv(os.path.join(pathdf, filedf))
        else:
            temp_tweets = pd.read_csv(os.path.join(pathdf, filedf))
            temp_tweets = pd.concat([df, temp_tweets])  # extract ignore index
            temp_tweets.to_csv(os.path.join(pathdf, filedf))

    def get_tweets_df(self, symbol: str, pathdf: str, filedf: str) -> pd.DataFrame():
        if not os.path.exists(os.path.join(pathdf, filedf)):
            print(f'This symbols {symbol} tweet not have')
            return
        else:
            temp_tweets = pd.read_csv(os.path.join(pathdf, filedf), index_col='index_col')
        return temp_tweets

    def throw_unnamed_cols(self, query: str, path_df, file_df) -> None:
        df_tweet = self.get_tweets_df(query, path_df, file_df)
        index_columns_list = ['created_at', 'text', 'source', 'name', 'username',
              'location', 'verified', 'description']
        df_tweet = df_tweet.drop([i for i in df_tweet.columns.to_list() if i not in index_columns_list], axis=1)
        df_tweet.reset_index(inplace=True)
        df_tweet['index_col'] = df_tweet.index
        df_tweet = df_tweet.set_index('index_col')        
        df_tweet.to_csv(os.path.join(path_df, file_df))