import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
import technical_analysis.backtest_kz as bt
from data_pipelines.data_manipulation import DataManipulation
import warnings
warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)

SYMBOL = 'SASA.IS'
scale = 1
range = [5, 10, 15, 20, 25, 30]
range = [i*scale for i in range]
period = '1y'
interval = '1d'
start_date = ''
end_data = ''
source = 'yahoo'

data = DataManipulation(SYMBOL, source, range, period=period, interval=interval, scale=scale, prefix_path='..')

df = data.df
print(df.head())

sample = df[['Open','High','Low','Close','Volume']].copy()

trade = bt.hisse_strategy_bt(df)
print(trade.head())

bt.bt_plot_strategy(df, trade)

trade_sheet = bt.bt_crossover(df, 'ema_5', 'ema_10')
bt.bt_plot_crossover(df, trade_sheet, 'ema_5', 'ema_10')

trade_sheet = bt.bt_threshold(df, 'cci_10', 220, 0)
bt.bt_plot_ind(df, trade_sheet, 'cci_15')