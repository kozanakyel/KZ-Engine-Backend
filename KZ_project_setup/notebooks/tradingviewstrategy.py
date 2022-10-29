import pandas as pd
from tradingview_ta import TA_Handler, Interval, Exchange
from tradingview_ta import *
import yfinance as yf
import time

exchange_bist = 'bist'
exchange_crypto = 'binance'
screener_bist = 'turkey'
screener_crypto = 'crypto'

# df = pd.read_csv('./symbol_data/crypto.csv', index_col='Order')
df = pd.read_csv('./symbol_data/ana_market.csv', index_col='Order')
df_symbols = df['Code'].to_list()
df_symbols = [f'{exchange_bist}:{i.lower()}' for i in df_symbols]

# for initial position -1, then we chack if condition satisfies convert 1, if not 0
df_symbols_dict = [(i.upper(), -1) for i in df_symbols] 
df_symbols_dict = dict(df_symbols_dict)
# return 'bist:sasa':-1

def indicator_strategy_buy_signal(stock_ind: dict, symbol_pair: dict, symbol: str, screener: str, exchange: str):
    # for current price we must get the 1 minute data otherwise our indicators wrong decision maybe
    # this method could be very slow for our app
    current_price = TA_Handler(  
        symbol=symbol.split(':')[1],
        screener=screener,
        exchange=exchange,
        interval=Interval.INTERVAL_1_MINUTE
        )
    current_price = current_price.get_analysis().indicators['close']

    pattern1 = stock_ind['EMA5'] >= stock_ind['SMA10']
    pattern2 = stock_ind['MACD.macd'] > stock_ind['MACD.signal']
    pattern3 = stock_ind['Ichimoku.BLine'] < current_price
    pattern4 = stock_ind['ADX+DI'] >= stock_ind['ADX-DI']
    pattern5 = stock_ind['SMA10'] < current_price
    pattern6 = stock_ind['Stoch.K'] > stock_ind['Stoch.D']

    all_pattern = (pattern1 and pattern2 and pattern3 and pattern4 and pattern5 and pattern6)

    if all_pattern and symbol_pair[symbol] == -1:
        print(f'INITLIAZE SIGNAL AT NOW BUY {symbol} at {current_price}')
        symbol_pair[symbol] = 1
    elif not all_pattern and symbol_pair[symbol] == -1:
        symbol_pair[symbol] = 0
    elif all_pattern and symbol_pair[symbol] == 0:
        print(f'BUY SIGNAL FOR {symbol} at {current_price}')
        symbol_pair[symbol] = 1
    elif symbol_pair[symbol] == 1 and not all_pattern:
        print(f'SELL SIGNAL FOR {symbol} at {current_price}')
        symbol_pair[symbol] = 0

repeat = 0
print(f'INTERVAL_1_HOUR {screener_bist} INDEX BUY/SELL Signals')
while True:
    analysis = get_multiple_analysis(
        screener=screener_bist, 
        interval=Interval.INTERVAL_1_DAY, 
        symbols=df_symbols
        )

    print(f'Time: {repeat*30} minute')
    repeat += 1
    
    for k in analysis:
        dict_ind = analysis[k].indicators
        # indicator_strategy_buy_signal(dict_ind, bist_symbols_dict, k, 'turkey', 'bist')
        indicator_strategy_buy_signal(dict_ind, df_symbols_dict, k, screener_bist, exchange_bist)
    time.sleep(60*30)


