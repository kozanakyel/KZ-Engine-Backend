import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas_ta as ta

from KZ_project.technical_analysis.indicators import Indicators
import KZ_project.technical_analysis.backtest_kz as bt
from KZ_project.ml_pipeline.data_generator.data_manipulation import DataManipulation
from KZ_project.strategy_kz.strategy_var import StrategyVar

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)


interval = 'hour'
df_m = pd.read_csv(f'./data/tweets_data/btc/btc_{interval}.csv', index_col=[0])
df_e = pd.read_csv(f'./data/tweets_data/eth/eth_{interval}.csv', index_col=[0])
df_k = pd.read_csv(f'./data/tweets_data/bnb/bnb_{interval}.csv', index_col=[0])
df_r = pd.read_csv(f'./data/tweets_data/ada/ada_{interval}.csv', index_col=[0])

df_com = df_m.copy()
df_com.columns = [f'btc']
df_com[f'eth'] = df_e['compound_total']
df_com[f'bnb'] = df_k['compound_total']
df_com[f'ada'] = df_r['compound_total']
df_com.dropna(inplace=True)
df_com.plot(figsize=(18,5), color=['y', 'b', 'r', 'k'])

SYMBOL = ''
scale = 1
range_list = [5, 9,10,12, 15, 20, 26, 50, 200]
range_list = [i*scale for i in range_list]
period = 'max'
interval = '1d'
start_date = ''
end_data = ''
source = 'yahoo'


res_dict = []
st = StrategyVar('emacross')
for i in ['TCELL.IS', 'BIMAS.IS', 'AKSA.IS']:   #, 'CCOLA.IS', 'CIMSA.IS', 'EREGL.IS', 'PETKM.IS'
    symbol = i
    data = DataManipulation(symbol, source, range_list, period=period, interval=interval, scale=scale, prefix_path='..', saved_to_csv=False)
    df_result = st.create_ema_labels(data.df.copy())
    res = st.find_best_cross_strategy(symbol, df_result)
    res_dict.append(res)
    
res_dict

symbols = pd.read_csv('./data/symbol_data/bist100.csv', index_col=[0])

for i in range(1, 3):
    rnd = random.randint(1, len(symbols))
    SYMBOL = f"{symbols.loc[rnd, 'Code']}.IS"
    data = DataManipulation(SYMBOL, source, range_list, period=period, interval=interval, scale=scale, prefix_path='..', saved_to_csv=False)

    df_temp = data.df.loc['2022-05':,:].copy()
    
    #SUPERTREND indicator
    sti = ta.supertrend(df_temp['high'], df_temp['low'], df_temp['close'], length=7, multiplier=3)
    sti.drop(columns=['SUPERTd_7_3.0', 'SUPERTl_7_3.0'], inplace=True)
    sti.columns = ['supertrend_support', 'supertrend_resistance']

    # PLOT THE EMA'S
    fig, ax1 = plt.subplots(1,1, figsize=(20,10))
    ax1.plot(df_temp.ema_9, label='ema_9')
    ax1.plot(df_temp.ema_12, label='ema_12')
    ax1.plot(df_temp.ema_20, label='ema_20')
    ax1.plot(df_temp.ema_26, label='ema_26')
    ax1.plot(df_temp.ema_50, label='ema_50')
    ax1.plot(df_temp.ema_200, label='ema_200')
    ax1.plot(df_temp.close, label='Close', linewidth=3)
    ax1.set_title(f'{SYMBOL}')
    ax1.legend()
    fig.savefig(f'./data/plots/ema_strategy_{SYMBOL}.png')

    # PLOT THE SUPERTREND
    fig, ax = plt.subplots(figsize=(20,7))
    ax.plot(sti)
    ax.plot(df_temp.close, label='Close')
    ax.set_title(f'{SYMBOL} SUPERTREND INDICATOR')
    ax.legend()
    fig.savefig(f'./data/plots/supertrend_{SYMBOL}.png')
    
