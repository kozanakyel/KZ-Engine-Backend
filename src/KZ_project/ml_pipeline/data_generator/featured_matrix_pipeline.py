from datetime import timedelta
import pandas as pd
from KZ_project.ml_pipeline.data_generator.data_checker import DataChecker
from KZ_project.ml_pipeline.data_generator.data_creator import DataCreator
from KZ_project.ml_pipeline.data_generator.feature_extractor import FeatureExtractor
from KZ_project.ml_pipeline.data_generator.file_data_checker import FileDataChecker
from KZ_project.ml_pipeline.indicators.factory_indicator_builder import FactoryIndicatorBuilder
from KZ_project.ml_pipeline.services.twitter_service.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.ml_pipeline.services.twitter_service.twitter_collection import TwitterCollection

"""
@author: Kozan Ugur AKYEL
@since: 07/08/2022

This file main purpose is that getting finance data,
Creating indicators column for Dataframe, 
Cleaning the data, is necessary or if you want
convert the data to featured extracted data with some specific strategies.
Add lags, add, indicators specific strategis and 
obtain nearly binary matrix, 
some strategies need 3 level layer for labelling.
But our main purposes is that matrix preparation for the
our Forecaster model...
""" 

class FeaturedMatrixPipeline():
    def __init__(self, data_creator: DataCreator, 
                 data_checker: DataChecker):
        self.data_creator = data_creator
        self.data_checker = data_checker
        
        
    @property
    def interval(self) -> str:
        return self.data_creator.interval
        
    def create_aggregate_featured_matrix(self):
        self.data_creator.df = self.data_creator.download_ohlc_from_client()
        self.data_creator.df = self.data_creator.create_datetime_index(self.data_creator.df)
        self.data_creator.df = self.data_creator.column_names_preparation(self.data_creator.df, self.data_creator.range_list)
        indicator_df_result = FactoryIndicatorBuilder.create_indicators_columns(self.data_creator.df, self.data_creator.range_list, logger=self.data_creator.logger)  
        self.data_creator.df = indicator_df_result
        self.data_creator.df = self.data_creator.reindex_and_sorted_cols(self.data_creator.df)
        feature_extractor = FeatureExtractor(self.data_creator.df, self.data_creator.range_list, self.data_creator.interval, self.data_creator.logger)
        feature_extractor.create_featured_matrix()
        self.agg_featured_matrix = feature_extractor.featured_matrix 
        print('finish agg')
        return self.agg_featured_matrix
        
        

    
class SentimentFeaturedMatrixPipeline(FeaturedMatrixPipeline):
    
    def __init__(self, data_creator: DataCreator, 
                 data_checker: DataChecker, hashtag, lang: str="en"):
        super().__init__(data_creator, data_checker)
        self.client_twitter = TwitterCollection()
        self.tsa = TweetSentimentAnalyzer(lang=lang)
        self.tweet_counts = 0
        self.hashtag = hashtag
        
        
    def create_sentiment_aggregate_feature_matrix(self):
        try:
            client_twt, tsa, daily, hourly, data = self.construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger(self.hashtag)
            hourly_tsa = self.get_tweet_sentiment_hourly(hourly)
            self.agg_sent_feature_matrix = self.composite_tweet_sentiment_and_data_manipulation(hourly_tsa, tsa)
        except Exception as e:
            print(f'Error for Tweepy: {e}')
            self.agg_sent_feature_matrix = super().create_aggregate_featured_matrix().copy()
            self.agg_sent_feature_matrix["twitter_sent_score"] = 0.0
            self.tweet_counts = 0
        print(self.agg_sent_feature_matrix["twitter_sent_score"])
        return self.agg_sent_feature_matrix
            
        
    def get_sentiment_daily_hourly_scores(self, hastag: str, 
                                           twitter_client: TwitterCollection,
                                           tsa: TweetSentimentAnalyzer,
                                           hour: int=24*7):
        df_tweets = twitter_client.get_tweets_with_interval(hastag, 'en', hour=hour, interval=int(self.interval[0]))
        self.tweet_counts = df_tweets.shape[0]
        path_df = f'./data/tweets_data/{hastag}/'
        daily_sents, hourly_sents = tsa.create_sent_results_df(hastag, df_tweets, path_df, saved=False)
        return daily_sents, hourly_sents 
    
    def construct_client_twt_tsa_daily_hourly_twt_datamanipulation_logger(self, hastag: str) -> tuple:

        client_twt, tsa = self.client_twitter, self.tsa
        daily, hourly = self.get_sentiment_daily_hourly_scores(hastag, client_twt, tsa)

        data = self.data_creator
        return client_twt, tsa, daily, hourly, data
    
    def get_tweet_sentiment_hourly(self, sent_tweets_hourly: pd.DataFrame()):
        sent_tweets = sent_tweets_hourly
        sent_tweets.Datetime = pd.to_datetime(sent_tweets.index)
        sent_tweets = sent_tweets.reset_index()
        sent_tweets.set_index('Datetime', inplace=True, drop=True)
        #sent_tweets.index = sent_tweets.index.tz_convert(None) # only hourly
        sent_tweets.index = sent_tweets.index + timedelta(hours=4)
        return sent_tweets
    
    def composite_tweet_sentiment_and_data_manipulation(self,
                                                         sent_tweets: pd.DataFrame(),
                                                         tsa: TweetSentimentAnalyzer):

        mtrix = super().create_aggregate_featured_matrix()
        df_price_ext = mtrix.copy()
        df_price_ext.index = df_price_ext.index + timedelta(hours=3)
        df_final = tsa.concat_ohlc_compound_score(df_price_ext, sent_tweets)
        del df_price_ext
        df_final = df_final.rename(columns={"compound_total":"twitter_sent_score"})
        print(df_final["twitter_sent_score"])
        df_final.dropna(inplace=True)
        return df_final 

        
