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

def create_retuns_data(X_pd):
        X_pd["strategy"] = X_pd.signal * X_pd["log_return"]
        X_pd[["log_return", "strategy"]].sum().apply(np.exp)
        
        X_pd["cstrategy"] = X_pd["strategy"].cumsum().apply(np.exp) 
        X_pd["creturns"] = X_pd.log_return.cumsum().apply(np.exp) 
        
def trade_fee_net_returns(X_pd: pd.DataFrame(), symbol):    
        X_pd["trades"] = X_pd.signal.diff().fillna(0).abs()    
        commissions = 0.00075 # reduced Binance commission 0.075%
        other = 0.0001 # proportional costs for bid-ask spread & slippage (more detailed analysis required!)
        ptc = np.log(1 - commissions) + np.log(1 - other)
    
        X_pd["strategy_net"] = X_pd.strategy + X_pd.trades * ptc # strategy returns net of costs
        X_pd["cstrategy_net"] = X_pd.strategy_net.cumsum().apply(np.exp)
    
        X_pd[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8),  title = f"{symbol} - Buy and Hold")
        plt.show()
        return X_pd[["creturns", "cstrategy", "cstrategy_net"]].to_json()

# conn = psycopg2.connect(get_postgres_uri())

summ = 0
for i in range(len(coin_lists)):
    # cur = conn.cursor()
    # cur.execute(f"SELECT * FROM signal_trackers WHERE ticker='{coin_lists[i]}'")
    # rows = cur.fetchall()

    # df = pd.DataFrame(rows, columns=["id","datetime_t","signal", "ticker", "tweet_counts", "forecast_model_id","created_at"])
    # df = df.set_index("datetime_t")
    
    # df.to_csv(f'{DB_RESULTS_PATH}/{coin_lists[i]}_signals.csv')
    
    # ohlc = client.get_history(coin_symbol[i], '1h', '2023-05-02')
    # pct_changes = ohlc['close'].pct_change()
    # ohlc.index = pd.to_datetime(ohlc.index) + timedelta(hours=3)
    # ohlc['log_return'] = log_returns
    # ohlc['binary_return'] = (ohlc['log_return'] > 0).astype(int)
    
    # ohlc.to_csv(f'{DB_RESULTS_PATH}/{coin_symbol[i]}_ohlc.csv')
    
    # print(f'result for: {df.signal} || {ohlc.binary_return}')
    df_s = pd.read_csv(f'{DB_RESULTS_PATH}/{coin_lists[i]}_signals.csv', index_col=["datetime_t"])
    df_o = pd.read_csv(f'{DB_RESULTS_PATH}/{coin_symbol[i]}_ohlc.csv', index_col=[0])
    df_s['change'] = df_o['close'].pct_change()
    df_s['log_return'] = df_o.ta.log_return()
    df_s['binary'] = (df_s['change'] > 0).astype(int)
    df_s['close'] = df_o['close']
    df_s['lasts'] = df_s['binary'] + df_s['signal']
    df_s = df_s.dropna()
    #summ = summ + sum(df_s['binary_return'] and df_s['signal'])
    print(f'{coin_symbol[i]} {df_s[["change", "close", "log_return","binary", "signal", "lasts"]]}')
    df_s.to_csv(f'{DB_RESULTS_PATH}/{coin_symbol[i]}_ohlc.csv')
    
    create_retuns_data(df_s)
    trade_fee_net_returns(df_s, coin_symbol[i])
    
    
    
