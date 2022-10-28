import pandas as pd
from tradingview_ta import TA_Handler, Interval, Exchange
from tradingview_ta import *
import time

df = pd.read_csv('./bist_symbol_data/ana_market.csv', index_col='Order')
bist_symbols = df['Code'].to_list()
bist_symbols = [f'bist:{i.lower()}' for i in bist_symbols]

# for initial position -1, then we chack if condition satisfies convert 1, if not 0
bist_symbols_dict = [(i.upper(),-1) for i in bist_symbols] 
bist_symbols_dict = dict(bist_symbols_dict)
# return 'bist:sasa':-1

def indicator_strategy_buy_signal(stock_ind: dict, symbol_pair: dict, symbol: str):

    pattern1 = stock_ind['EMA5'] >= stock_ind['SMA10']
    pattern2 = stock_ind['MACD.macd'] > stock_ind['MACD.signal']
    pattern3 = stock_ind['Ichimoku.BLine'] < stock_ind['close']
    pattern4 = stock_ind['ADX+DI'] >= stock_ind['ADX-DI']
    pattern5 = stock_ind['SMA10'] < stock_ind['close']
    pattern6 = stock_ind['Stoch.K'] > stock_ind['Stoch.D']

    all_pattern = (pattern1 and pattern2 and pattern3 and pattern4 and pattern5 and pattern6)

    if all_pattern and symbol_pair[symbol] == 0:
        print(f'BUY SIGNAL FOR {symbol} at {stock_ind["close"]}')
        symbol_pair[symbol] = 1
    elif all_pattern and symbol_pair[symbol] == -1:
        symbol_pair[symbol] = 1
    elif not all_pattern and symbol_pair[symbol] == 1:
        print(f'SELL SIGNAL FOR {symbol} at {stock_ind["close"]}')
        symbol_pair[symbol] = 0
    else:
        symbol_pair[symbol] = 0

repeat = 0
while True:
    analysis = get_multiple_analysis(
        screener="turkey", 
        interval=Interval.INTERVAL_15_MINUTES, 
        symbols=bist_symbols
        )
    repeat += 1
    print(f'Repeat number: {repeat}')
    for k in analysis:
        dict_ind = analysis[k].indicators
        indicator_strategy_buy_signal(dict_ind, bist_symbols_dict, k)
    time.sleep(60*10)


