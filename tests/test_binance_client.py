import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from KZ_project.binance_domain.binance_client import BinanceClient
from KZ_project.binance_domain.ai_trader import AITrader
from KZ_project.twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.data_pipelines.data_manipulation import DataManipulation
from KZ_project.dl_models.xgboost_forecaster import XgboostForecaster
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
    trader = AITrader("BNBUSDT", "1m", client)
    trader.start_trading()
    time.sleep(60*3)
    trader.stop_trading()
    
test_web_socket_bnb()


