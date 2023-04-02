import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from KZ_project.ml_pipeline.services.binance_service.binance_client import BinanceClient
from KZ_project.ml_pipeline.services.binance_service.ai_trader import AITrader
from KZ_project.ml_pipeline.services.twitter_service.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.ml_pipeline.data_generator.data_manipulation import DataManipulation
from KZ_project.ai_model_creator.xgboost_forecaster import XgboostForecaster
from KZ_project.logger.logger import Logger
import numpy as np
import matplotlib.pyplot as plt

from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import time

import config

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret_key = os.getenv('BINANCE_SECRET_KEY')

client = BinanceClient(api_key, api_secret_key)

def test_get_history():
    now = datetime.utcnow()
    dff = client.get_history(symbol = "BTCUSDT", interval = "1h", start = str(now - timedelta(hours = 2)), end=None)
    print(dff)

def test_web_socket_bnb():
    trader_btc = AITrader(symbol="BTCUSDT", name="btc", bar_length="1m", client=client, units=20)
    trader_bnb = AITrader(symbol="BNBUSDT", name="bnb", bar_length="1m", client=client, units=20)
    trader_btc.start_trading()
    trader_bnb.start_trading()
    time.sleep(60*20)
    trader_btc.stop_trading()
    
test_web_socket_bnb()


