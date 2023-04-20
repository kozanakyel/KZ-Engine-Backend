import pandas as pd
#from binance.client import Client
#import pandas as ta
import numpy as np
import matplotlib.pyplot as plt

from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
#import requests

from KZ_project.Infrastructure.logger.logger import Logger
#from KZ_project.core.adapters.forecastmodel_repository import ForecastModelRepository
#from KZ_project.ml_pipeline.services.twitter_service.twitter_collection import TwitterCollection
#from KZ_project.ml_pipeline.services.twitter_service.tweet_sentiment_analyzer import TweetSentimentAnalyzer
#from KZ_project.ml_pipeline.data_generator.data_manipulation import DataManipulation
#from KZ_project.ml_pipeline.ai_model_creator.xgboost_forecaster import XgboostForecaster
from KZ_project.ml_pipeline.services.binance_service.binance_client import BinanceClient
from KZ_project.ml_pipeline.services.binance_service.forecast_engine_hourly import ForecastEngineHourly

import KZ_project.Infrastructure.config as config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

#from KZ_project.webapi.services import services

#orm.start_mappers()
engine = create_engine(config.get_postgres_uri())
get_session = sessionmaker(bind=engine)


class AITrader():
    
    def __init__(self, symbol: str, name: str, bar_length, client: BinanceClient, units, engine: ForecastEngineHourly , logger: Logger=None):
        self.symbol = symbol
        self.name = name
        self.bar_length = bar_length
        self.data = pd.DataFrame(columns = ["Open", "High", "Low", "Close", "Volume", "Complete"])
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]   
        self.client = client
        self.postion = 0
        self.trades = 0
        self.trade_values = []
        self.units = units
        self.logger = logger
        self.engine = engine
      
    def start_trading(self):
        
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length)
        # "else" to be added later in the course 
    

    
    def stream_candles(self, msg):
        
        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit = "ms") +  timedelta(hours = 3)
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms") +  timedelta(hours = 3)
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]
         
        start_date = (event_time - timedelta(days=10)).strftime('%Y-%m-%d')
    
        # print out
        print("Time: {} | Price: {} | Complete: {} | Symbol {}".format(event_time, close, complete, self.symbol))
        
        if complete:
            #print(f'Write logic for prediction in this area')
            #print(f'start date: {start_date} and type {type(start_date)}')
            
            ai_type, Xt, next_candle_prediction = self.engine.forecast_builder(start_date=start_date)
            
            #self.main_prediction(start_date=start_date, name=self.name, symbol=self.symbol)
            #self.execute_trades()
        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low, close, volume, complete]
        
    def stop_trading(self):
        self.twm.stop()

    def execute_trades(self): 
        if self.prepared_data["position"].iloc[-1] == 1: # if position is long -> go/stay long
            if self.position == 0:
                order = self.client.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG")  # NEW
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0: # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = self.client.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL")  # NEW
            self.position = 0
    
    def report_trade(self, order, going):  # NEW
        
        # extract data from order object
        side = order["side"]
        time = pd.to_datetime(order["transactTime"], unit = "ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)
        
        # calculate trading profits
        self.trades += 1
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units) 
        
        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]), 3) 
            cum_profits = round(np.sum(self.trade_values), 3)
        else: 
            real_profit = 0
            cum_profits = round(np.sum(self.trade_values[:-1]), 3)
        
        # print trade report
        print(2 * "\n" + 100* "-")
        print("{} | {}".format(time, going)) 
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(time, base_units, quote_units, price))
        print("{} | Profit = {} | CumProfits = {} ".format(time, real_profit, cum_profits))
        print(100 * "-" + "\n")
        
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)       
        
   
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    from datetime import datetime, timedelta
    from KZ_project.ml_pipeline.services.binance_service.binance_client import BinanceClient
    from KZ_project.ml_pipeline.services.binance_service.ai_trader import AITrader
    import time

    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret_key = os.getenv('BINANCE_SECRET_KEY')

    client = BinanceClient(api_key, api_secret_key) 

    def get_history():
        now = datetime.utcnow()
        dff = client.get_history(symbol = "BTCUSDT", interval = "1h", start = str(now - timedelta(hours = 2)), end=None)
        print(dff)


    def web_socket_trader_starter():
        cr_list = [{"symbol":"BTCUSDT", "name":"btc", "bar_length":"3m"}, 
                   {"symbol":"BNBUSDT", "name":"bnb", "bar_length":"3m"},
                   {"symbol":"XRPUSDT", "name":"xrp", "bar_length":"5m"},
                   {"symbol":"ETHUSDT", "name":"eth", "bar_length":"5m"},
                   {"symbol":"DOGEUSDT", "name":"doge", "bar_length":"5m"}]
        trader_c_list = []
        for coin_d in cr_list:
            engine = ForecastEngineHourly(coin_d["name"], coin_d["symbol"])
        
            trader_coin_d = AITrader(
                symbol=coin_d["symbol"], 
                name=coin_d["name"], 
                bar_length=coin_d["bar_length"], 
                client=client, 
                units=20, 
                engine=engine
                )
            
            trader_c_list.append(trader_coin_d)
        
        for i in trader_c_list:
            i.start_trading()    
        
        time.sleep(60*20)
        
        for i in trader_c_list:
            i.stop_trading()
            
    def web_socket_trader_starter_btc():
        engine = ForecastEngineHourly("btc", "BTCUSDT")
        
        trader_coin_d = AITrader(
                symbol="BTCUSDT", 
                name="btc", 
                bar_length="3m", 
                client=client, 
                units=20, 
                engine=engine
                ) 
        trader_coin_d.start_trading()   
    
    web_socket_trader_starter()
    
    