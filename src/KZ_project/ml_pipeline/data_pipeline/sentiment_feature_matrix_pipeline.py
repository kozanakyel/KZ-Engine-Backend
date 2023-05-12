from datetime import timedelta
import pandas as pd
from KZ_project.Infrastructure.file_processor.data_checker import DataChecker
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
from KZ_project.ml_pipeline.data_pipeline.featured_matrix_pipeline import FeaturedMatrixPipeline
from KZ_project.Infrastructure.file_processor.file_data_checker import FileDataChecker
from KZ_project.ml_pipeline.sentiment_analyzer.sentiment_analyzer import SentimentAnalyzer
from KZ_project.ml_pipeline.services.twitter_service.twitter_collection import TwitterCollection


class SentimentFeaturedMatrixPipeline(FeaturedMatrixPipeline):
    
    def __init__(
        self, 
        data_creator: DataCreator, 
        data_checker: DataChecker, 
        hashtag: str,         # use only ticker like the; btc, xrp, and also look coingecko name 
        lang: str="en", 
        is_twitter: bool=True
    ):
        super().__init__(data_creator, data_checker)
        self.tweet_counts = 0
        self.hashtag = hashtag
        self.is_twitter = is_twitter
        if is_twitter:
            self.client_twitter = TwitterCollection()
            self.tsa = SentimentAnalyzer(lang=lang)
        
    def create_sentiment_aggregate_feature_matrix(self) -> pd.DataFrame():
        if self.is_twitter:
            try:
                self.agg_sent_feature_matrix = self.create_matrix_with_twitter()
            except Exception as e:
                print(f'Error for Tweepy: {e}')
                self.agg_sent_feature_matrix = self.create_matrix_without_twitter()
            return self.agg_sent_feature_matrix
        else:
            self.agg_sent_feature_matrix = self.create_matrix_without_twitter()
            return self.agg_sent_feature_matrix
    
    def create_matrix_without_twitter(self):
        agg_sent_feature_matrix = super().create_aggregate_featured_matrix().copy()
        agg_sent_feature_matrix["twitter_sent_score"] = 0.0
        # for binance exchange time module
        # if self.data_creator.interval[-1] == 'h':   # if interval hourly otherwise daily
        agg_sent_feature_matrix.index = agg_sent_feature_matrix.index + timedelta(hours=3)
        self.tweet_counts = 0
        return agg_sent_feature_matrix
    
    def create_matrix_with_twitter(self):
        client_twt, tsa, daily, hourly, data = self.construct_client_twt_tsa_hourly_twt_datamanipulation_logger(self.hashtag)
        hourly_tsa = self.get_tweet_sentiment_hourly(hourly)
        agg_sent_feature_matrix = self.composite_tweet_sentiment_and_data_manipulation(hourly_tsa, tsa)
        return agg_sent_feature_matrix
            
        
    def get_sentiment_hourly_scores(
        self, hastag: str, 
        twitter_client: TwitterCollection,
        tsa: SentimentAnalyzer,
        hour: int=24*7
    ):
        df_tweets = twitter_client.get_tweets_with_interval(hastag, 'en', hour=hour, interval=int(self.interval[0]))
        self.tweet_counts = df_tweets.shape[0]
        path_df = f'./data/tweets_data/{hastag}/'
        daily_sents, hourly_sents = tsa.create_sent_results_df(hastag, df_tweets, path_df, saved=False)
        return daily_sents, hourly_sents 
    
    def construct_client_twt_tsa_hourly_twt_datamanipulation_logger(self, hastag: str) -> tuple:
        """
        Review code this method is not necessary refactor and extract from this class!!!!
        """
        client_twt, tsa = self.client_twitter, self.tsa
        daily, hourly = self.get_sentiment_hourly_scores(hastag, client_twt, tsa)

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
                                                         tsa: SentimentAnalyzer):

        mtrix = super().create_aggregate_featured_matrix()
        df_price_ext = mtrix.copy()
        df_price_ext.index = df_price_ext.index + timedelta(hours=3)
        print(f'dfindex: {df_price_ext.index[-1]} and teweet index: {sent_tweets.index[-1]}')
        if df_price_ext.index[-1] != sent_tweets.index[-1]:
            sent_tweets.loc[df_price_ext.index[-1]] = 0.0
            print(f'Lastteweet index: {sent_tweets.index[-1]}')
        df_final = tsa.concat_ohlc_compound_score(df_price_ext, sent_tweets)
        del df_price_ext
        df_final = df_final.rename(columns={"compound_total":"twitter_sent_score"})
        # print(df_final["twitter_sent_score"])
        df_final.dropna(inplace=True)
        return df_final 
    
    
if __name__ == '__main__':
    from KZ_project.ml_pipeline.services.yahoo_service.yahoo_client import YahooClient
    MAIN_PATH = '/data/outputs/data_ind/'
    PURE_PATH = '/data/pure_data/'
    FEATURE_PATH = '/data/outputs/feature_data/'
    PREFIX_PATH = '.'
    SYMBOL = 'BTC-USD' 
    PERIOD = "2y"
    INTERVAL = '1d'
    START_DATE = '2021-06-30'
    END_DATE = '2022-07-01'
    HASHTAG = 'btc'
    scale = 1
    range_list = [i for i in range(5,21)]
    range_list = [i*scale for i in range_list]
    source='yahoo'

    f = FileDataChecker(symbol=SYMBOL, period=PERIOD, interval=INTERVAL, 
                    start_date=START_DATE, end_date=END_DATE, prefix_path=PREFIX_PATH, 
                    main_path=MAIN_PATH, pure_path=PURE_PATH,
                    feature_path=FEATURE_PATH)

    y = YahooClient()
    d = DataCreator(symbol=SYMBOL, source=source, range_list=range_list, period=PERIOD, interval=INTERVAL, 
                    start_date=START_DATE, end_date=END_DATE, 
                    data_checker=f,
                    client=y)  
    
    agg_matrix = SentimentFeaturedMatrixPipeline(d, None, HASHTAG, is_twitter=False)
    amtrix = agg_matrix.create_sentiment_aggregate_feature_matrix()
    print(amtrix)
