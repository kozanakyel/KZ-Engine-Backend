from datetime import timedelta
import math
from matplotlib import pyplot as plt
import psycopg2
import pandas as pd
import os
import numpy as np
import pandas_ta as ta

from KZ_project.Infrastructure.config import get_postgres_uri
from KZ_project.ml_pipeline.services.binance_service.binance_client import BinanceClient
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret_key = os.getenv('BINANCE_SECRET_KEY')

client = BinanceClient(api_key, api_secret_key) 

DB_RESULTS_PATH = "src/KZ_project/Infrastructure/backtesting/db_results"

coin_lists = ['btc', 'xrp', 'eth', 'doge', 'bnb']
coin_symbol = ['BTCUSDT', 'XRPUSDT', 'ETHUSDT', 'DOGEUSDT', 'BNBUSDT']

conn = psycopg2.connect(get_postgres_uri())

for i in range(len(coin_lists)):
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM forecast_models WHERE hashtag='{coin_lists[i]}'")
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["id","symbol","source", "feature_counts", "model_name", 
                                     "interval", "ai_type", "hashtag", 
                                     "accuracy_score", "crypto_id", "created_at"])
    result_ohlc = pd.read_csv(f'{DB_RESULTS_PATH}/result_{coin_symbol[i]}_ohlc.csv', index_col="datetime_t")
    print(result_ohlc)
    ohlc_index = result_ohlc.index
    #merged_df = pd.merge(result_ohlc, df, left_on='forecast_model_id', right_on="id", how='inner')
    #merged_df = merged_df.set_index(ohlc_index)
    # merged_df.to_csv(f'{DB_RESULTS_PATH}/result_{coin_symbol[i]}.csv')
    
    print(df)
    
    