from datetime import timedelta
import psycopg2
import pandas as pd
import os
import numpy as np

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
    cur.execute(f"SELECT * FROM signal_trackers WHERE ticker='{coin_lists[i]}'")
    rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["id","datetime_t","signal", "ticker", "tweet_counts", "forecast_model_id","created_at"])
    df = df.set_index("datetime_t")
    
    df.to_csv(f'{DB_RESULTS_PATH}/{coin_lists[i]}_signals.csv')
    
    ohlc = client.get_history(coin_symbol[i], '1h', '2023-05-02')
    pct_changes = ohlc['close'].pct_change()
    log_returns = np.log(1 + pct_changes)
    ohlc.index = pd.to_datetime(ohlc.index) + timedelta(hours=3)
    ohlc['log_return'] = log_returns
    ohlc['binary_return'] = (ohlc['log_return'] > 0).astype(int)
    
    ohlc.to_csv(f'{DB_RESULTS_PATH}/{coin_symbol[i]}_ohlc.csv')
    
    print(f'result for: {df.signal} || {ohlc.binary_return}')
    
    
