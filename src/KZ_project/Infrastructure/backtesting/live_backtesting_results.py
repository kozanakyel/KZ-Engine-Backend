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
     X_pd["strategy"] = X_pd.signal.shift(1) * X_pd["log_return"]
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
    
    ohlc = client.get_history(coin_symbol[i], '1h', '2023-05-02')
    pct_changes = ohlc['close'].pct_change()
    ohlc.index = pd.to_datetime(ohlc.index) + timedelta(hours=3)
     
    ohlc.to_csv(f'{DB_RESULTS_PATH}/{coin_symbol[i]}_ohlc.csv')
     
    df_s = pd.read_csv(f'{DB_RESULTS_PATH}/{coin_lists[i]}_signals.csv', index_col=["datetime_t"])
    df_o = pd.read_csv(f'{DB_RESULTS_PATH}/{coin_symbol[i]}_ohlc.csv', index_col=[0])
    result = df_s.merge(df_o, how='inner', left_index=True, right_index=True)
    result['change'] = result['close'].pct_change()
    result['log_return'] = result.ta.log_return()
    result['binary'] = (result['change'] > 0).astype(int)
    result['lasts_results'] = result['binary'] + result['signal']
    result = result.dropna()
    # summ = summ + sum(df_s['binary_return'] and df_s['signal'])
    print(f'{coin_symbol[i]} {result}')
    # result = df_s.merge(df_o, how='inner', left_index=True, right_index=True)
    # result = result.merge(df2, how='inner', on='a')
    result.to_csv(f'{DB_RESULTS_PATH}/result_{coin_symbol[i]}_ohlc.csv', index_label="datetime_t")
    
    create_retuns_data(result)
    trade_fee_net_returns(result, coin_symbol[i])
    
    
    
