import pandas as pd
import numpy as np

from binance import ThreadedWebsocketManager
from datetime import timedelta

from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
from KZ_project.Infrastructure.services.binance_service.binance_client import BinanceClient
from KZ_project.ml_pipeline.ai_model_creator.engines.forecast_engine import ForecastEngine


class AITrackerService:

    def __init__(
            self,
            symbol: str,
            ticker: str,
            bar_length,
            client: BinanceClient,
            units,
            interval: str,
            logger: Logger = None,
            is_twitter: bool = True
    ):
        self.symbol = symbol
        self.ticker = ticker
        self.bar_length = bar_length
        self.data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Complete"])
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d",
                                    "1w", "1M"]
        self.client = client
        self.position = 0
        self.trades = 0
        self.trade_values = []
        self.units = units
        self.logger = logger
        self.interval = interval
        self.is_twitter = is_twitter



    def start_trading(self):

        self.twm = ThreadedWebsocketManager()
        self.twm.start()

        if self.bar_length in self.available_intervals:
            self.twm.start_kline_socket(callback=self.stream_candles,
                                        symbol=self.symbol, interval=self.bar_length)

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

        if self.interval[-1] == 'h':
            start_date = (event_time - timedelta(days=15)).strftime('%Y-%m-%d')
        elif self.interval[-1] == 'd':
            start_date = (event_time - timedelta(days=365)).strftime('%Y-%m-%d')

        # print out
        # print("Time: {} | Price: {} | Complete: {} | Symbol {}".format(event_time, close, complete, self.symbol))

        if complete:
            data_creator = DataCreator(
                symbol=self.symbol,
                source='binance',
                range_list=[i for i in range(5, 21)],
                period=None,
                interval=self.interval,
                start_date=start_date,
                client=self.client
            )
            sentiment_pipeline = ForecastEngine(
                data_creator,
                self.ticker,
                is_twitter=self.is_twitter
            )
            ai_type, Xt, next_candle_prediction = sentiment_pipeline.forecast_builder()
            print(f'prediction {Xt} {next_candle_prediction}')
            # self.execute_trades()
        # feed df (add new bar / update latest bar)
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


    def web_socket_trader_starter():
        interval = '1d'
        cr_list = [
            {"symbol": "BTCUSDT", "name": "btc", "bar_length": "5m"},
            {"symbol": "BNBUSDT", "name": "bnb", "bar_length": "5m"},
            {"symbol": "ETHUSDT", "name": "eth", "bar_length": "5m"},
        ]
        trader_c_list = []
        for coin_d in cr_list:
            trader_coin_d = AITrackerService(
                symbol=coin_d["symbol"],
                ticker=coin_d["name"],
                bar_length=coin_d["bar_length"],
                client=client,
                units=20,
                interval=interval,
                is_twitter=False
            )

            trader_c_list.append(trader_coin_d)

        for i in trader_c_list:
            i.start_trading()


    web_socket_trader_starter()
