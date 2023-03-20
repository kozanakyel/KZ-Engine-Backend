import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from KZ_project.binance_domain.binance_client import BinanceClient
from KZ_project.twitter.tweet_sentiment_analyzer import TweetSentimentAnalyzer
from KZ_project.data_pipelines.data_manipulation import DataManipulation
from KZ_project.dl_models.xgboost_forecaster import XgboostForecaster
from KZ_project.logger.logger import Logger
import numpy as np
import matplotlib.pyplot as plt

import config

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret_key = os.getenv('BINANCE_SECRET_KEY')

client = BinanceClient(api_key, api_secret_key)

def test_get_history():
    now = datetime.utcnow()
    dff = client.get_history(symbol = "BTCUSDT", interval = "1h", start = str(now - timedelta(hours = 2)), end=None)
    print(dff)

def test_websocket():
    client.start_ws()
    client.start_mini_ticker(callback=client.stream_data, symbol="BTCUSDT")
    client.stop_ws()

data = DataManipulation(config.YahooConfig.SYMBOL, config.YahooConfig.source, 
                        config.YahooConfig.range_list, period="1y", start_date=config.YahooConfig.start_date, 
                        end_date=config.YahooConfig.end_date, interval=config.YahooConfig.interval, 
                        scale=config.YahooConfig.SCALE, 
                        prefix_path='..', saved_to_csv=False,
                        logger=config.YahooConfig.logger)
df_price_ext = data.extract_features()

print(df_price_ext[["feature_label", "log_return"]])

df_price_ext["strategy"] = df_price_ext.feature_label.shift(1) * df_price_ext["log_return"]
df_price_ext[["log_return", "strategy"]].sum().apply(np.exp)
df_price_ext["cstrategy"] = df_price_ext["strategy"].cumsum().apply(np.exp) 
df_price_ext["creturns"] = df_price_ext.log_return.cumsum().apply(np.exp) 

df_price_ext.creturns.plot(figsize = (12, 8), title = "BTC/USDT - Buy and Hold", fontsize = 12)
plt.show()

df_price_ext[["creturns", "cstrategy"]].plot(figsize = (12 , 8), fontsize = 12)
plt.show()

