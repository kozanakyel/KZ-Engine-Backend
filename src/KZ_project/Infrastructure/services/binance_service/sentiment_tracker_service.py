import pandas as pd
from binance import ThreadedWebsocketManager

from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.Infrastructure.services.binance_service.binance_client import BinanceClient
from KZ_project.Infrastructure.services.twitter_service.twitter_collection import TwitterCollection

from KZ_project.ml_pipeline.sentiment_analyzer.sentiment_analyzer import SentimentAnalyzer
import matplotlib.pyplot as plt
from KZ_project.webapi.services.services import add_sentiment_record_from_dataframe, get_all_sentiment_records
from KZ_project.webapi.entrypoints.flask_app import get_session


class SentimentTrackerService:

    def __init__(
            self,
            bar_length,
            client: BinanceClient,
            interval: str,
            logger: Logger = None,
            twm: ThreadedWebsocketManager = None
    ):
        self.twm = twm
        self.bar_length = bar_length
        self.data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Complete"])
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d",
                                    "1w", "1M"]
        self.client = client
        self.logger = logger
        self.interval = interval

    def start_trading(self):

        # self.twm.start()

        if self.bar_length in self.available_intervals:
            self.twm.start_kline_socket(callback=self.stream_candles, symbol="BTCUSDT", interval=self.bar_length)

    def stream_candles(self, msg):

        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit="ms") + timedelta(hours=3)
        start_time = pd.to_datetime(msg["k"]["t"], unit="ms") + timedelta(hours=3)
        first = float(msg["k"]["o"])
        high = float(msg["k"]["h"])
        low = float(msg["k"]["l"])
        close = float(msg["k"]["c"])
        volume = float(msg["k"]["v"])
        complete = msg["k"]["x"]

        if complete:
            client = TwitterCollection()
            df = client.get_tweet_contents(tw_counts_points=10)
            sid = SentimentAnalyzer()
            df = sid.cleaning_tweet_data(df)
            df = sid.preprocessing_tweet_datetime(df)
            df = sid.get_sentiment_scores(df)
            sid.add_datetime_to_col(df)
            sent_scores = sid.get_sent_with_mean_interval(df, self.interval)
            # last_month = client.get_last_mont_df(sent_scores)  # uses dataframe to series procedure            
            # sent_scores = sent_scores.to_frame()   # dont forget the convert dataframe
            add_sentiment_record_from_dataframe(sent_scores, get_session())
        self.data.loc[start_time] = [first, high, low, close, volume, complete]

    def stop_trading(self):
        self.twm.stop()

    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    from datetime import timedelta
    from KZ_project.Infrastructure.services.binance_service.binance_client import BinanceClient

    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret_key = os.getenv('BINANCE_SECRET_KEY')

    client = BinanceClient(api_key, api_secret_key)
    twm = ThreadedWebsocketManager()    # for avoid thread sockets errors
    twm.start()
    interval = '1h'
    sentiment_tracker = SentimentTrackerService(
                bar_length='5m',
                client=client,
                interval=interval,
                twm=twm
            )
    sentiment_tracker.start_trading()