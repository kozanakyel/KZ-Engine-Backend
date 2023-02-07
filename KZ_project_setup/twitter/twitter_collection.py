import tweepy, datetime
from tweepy import OAuthHandler
import tweepy
import pandas as pd
import os
from dotenv import load_dotenv
import datetime
from datetime import timedelta as td
from logger.logger import Logger
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

access_token=os.getenv('TW_access_token')
access_token_secret=os.getenv('TW_access_token_secret')
consumer_key=os.getenv('TW_consumer_key')
consumer_secret=os.getenv('TW_consumer_secret')
MY_BEARER_TOKEN=os.getenv('TW_BEARER_TOKEN')

class TwitterCollection():
    def __init__(self, access_token=access_token, access_token_secret=access_token_secret,
                    consumer_key=consumer_key, consumer_secret=consumer_secret, 
                    bearer_token=MY_BEARER_TOKEN, logger: Logger=None, connection: bool=True):
        self.access_token = access_token
        self.access_token_secret=access_token_secret
        self.consumer_key=consumer_key
        self.consumer_secret=consumer_secret
        self.bearer_token=bearer_token
        self.logger = logger
        self.connect = connection
        if connection:
            self.connect_twitter()
            self.client = tweepy.Client(bearer_token=self.bearer_token)

    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)

    def connect_twitter(self):
        try:
            auth = OAuthHandler(self.consumer_key, self.consumer_secret)
            auth.set_access_token(self.access_token, self.access_token_secret)
            api = tweepy.API(auth, wait_on_rate_limit=True)
            self.log("Authentication Successfull")
        except:
            self.log("Error: Authentication Failed")

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


    def get_tweets_with_interval(self, hashtag: str, lang: str, start_time=None, finish_time=None, hour=24, interval = 1) -> pd.DataFrame():
        now = datetime.datetime.now(datetime.timezone.utc)
        if start_time == None:
            start_time = now - td(hours=hour)
        if finish_time == None:
            finish_time = now
        end_time = start_time + td(hours=interval)

        self.log(f'For hashtag {hashtag.upper()} with language {lang.upper()} start time: {start_time}, end time: {start_time + td(hours=hour)}')
        result_tweets = pd.DataFrame()
        while end_time <= finish_time - td(hours=interval):
            temp_tweets = self.get_tweets(hashtag, lang, start_time.isoformat(), end_time.isoformat())
            df_temp_tweets = self.converts_tweets_pd(temp_tweets)
            result_tweets = pd.concat([df_temp_tweets, result_tweets], ignore_index=True)
            start_time = start_time + td(hours=interval)
            end_time = end_time + td(hours=interval)
        self.log(f'For hashtag {hashtag.upper()} tweeets colected')
        return result_tweets

    def cleaning_tweet_data(self, df: pd.DataFrame()):
        df_tweets = df.copy()
        df_tweets.dropna(inplace=True)
        if 'Unnamed: 0' in df_tweets.columns:
            df_tweets.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
        df_tweets.drop(columns=['source', 'name', 'location', 'verified', 'description'], axis=1, inplace=True)
        blanks = []  # start with an empty list

        for i, created_at, text, *others in df_tweets.itertuples():  
            if type(text)==str:            
                if text.isspace():         
                    blanks.append(i)    
    
        df_tweets.drop(blanks, inplace=True)

    def write_tweets_csv(self, df: pd.DataFrame(), pathdf: str, filedf: str) -> None:
        if not os.path.exists(os.path.join(pathdf, filedf)):
            os.makedirs(pathdf, exist_ok=True)
            with open(os.path.join(pathdf, filedf), mode='a'): pass
            print
            df.to_csv(os.path.join(pathdf, filedf))
            self.log(f'Tweeets writes to File to first time: {pathdf} {filedf}')
        else:
            chunksize = 1000
            list_of_dataframes = []
            for df_read in pd.read_csv(os.path.join(pathdf, filedf), chunksize=chunksize, index_col=[0]):
                list_of_dataframes.append(df_read)
            temp_tweets = pd.concat(list_of_dataframes)
            self.log(f'Read Tweets from File and Chunksized to {chunksize}')
            temp_tweets = pd.concat([df, temp_tweets])
            self.cleaning_tweet_data(temp_tweets)
            temp_tweets.to_csv(os.path.join(pathdf, filedf))
            self.log(f'Concantenated tweets writed to file {pathdf} {filedf}')

    def get_tweets_df(self, symbol: str, pathdf: str, filedf: str) -> pd.DataFrame():
        if not os.path.exists(os.path.join(pathdf, filedf)):
            self.log(f'This symbols {symbol} tweet not have')
            return
        else:
            chunksized = 100000
            list_of_dataframes = []
            for df in pd.read_csv(os.path.join(pathdf, filedf), chunksize=chunksized, index_col=0, lineterminator='\n'):
                list_of_dataframes.append(df)
            temp_tweets = pd.concat(list_of_dataframes)
        return temp_tweets

    def throw_unnamed_cols(self, df_tweet) -> pd.DataFrame():
        index_columns_list = ['created_at', 'text', 'source', 'name', 'username',
              'location', 'verified', 'description']
        df_tweet = df_tweet.drop([i for i in df_tweet.columns.to_list() if i not in index_columns_list], axis=1)
        df_tweet.reset_index(inplace=True)
        if 'index' in df_tweet.columns:
            df_tweet.drop(columns=['index'], axis=1, inplace=True)
        return df_tweet