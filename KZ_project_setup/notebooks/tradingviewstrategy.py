import pandas as pd
from tradingview_ta import TA_Handler, Interval, Exchange
import tradingview_ta
from tradingview_ta import *

analysis = get_multiple_analysis(
        screener="turkey", 
        interval=Interval.INTERVAL_1_WEEK, 
        symbols=["bist:sasa", "bist:orma", "bist:sumas", "bist:aksa", "bist:petkm", "bist:yonga"]
        )

# EMA5 >= SMA10, MACD.macd > MACD.signal, Ichimoku.BLine < close, ADX+DI >= ADX-DI, SMA10 < close, Stoch.K > Stoch.D 
def indicator_strategy_1(stock_ind: dict, symbol: str):
    pattern1 = stock_ind['EMA5'] >= stock_ind['SMA10']
    pattern2 = stock_ind['MACD.macd'] > stock_ind['MACD.signal']
    pattern3 = stock_ind['Ichimoku.BLine'] < stock_ind['close']
    pattern4 = stock_ind['ADX+DI'] >= stock_ind['ADX-DI']
    pattern5 = stock_ind['SMA10'] < stock_ind['close']
    pattern6 = stock_ind['Stoch.K'] > stock_ind['Stoch.D']
    if(pattern1 and pattern2 and pattern3 and pattern4 and pattern5 and pattern6):
        print(f'Buy the stock {symbol}')

for k in analysis:
    dict_ind = analysis[k].indicators
    indicator_strategy_1(dict_ind, k)
