import pandas as pd
import numpy as np
import os
from technical_analysis.indicators import Indicators
import yfinance as yf
from technical_analysis.candlestick_features import *


class DataManipulation():
    def __init__(self, symbol: str, source: str, range: list, period=None, interval=None, 
                                start_date=None, end_date=None, scale=1, prefix_path='.'):
        self.symbol = symbol
        self.source = source
        self.scale = scale
        self.range = [i*scale for i in range]
        self.period = period
        self.interval = interval 
        self.start_date = start_date
        self.end_date = end_date
        self.prefix_path = prefix_path
        self.df = self.create_data_one(self.symbol, self.source, self.period, self.interval, prefix_path=self.prefix_path)

    
    def create_data_one(self, symbol, source, period=None, interval=None, prefix_path='.') -> pd.DataFrame():
        path_df = prefix_path+'/data/outputs/data_ind/'+symbol 
        pure_data = prefix_path+'/data/pure_data/'+symbol
        pure_file = f'{symbol}_{period}_{interval}.csv'
        file = f'{symbol}_df_{period}_{interval}.csv'
        
        if os.path.exists(os.path.join(path_df, file)):
            self.df = pd.read_csv(os.path.join(path_df, file))
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
            self.df = self.df.set_index('Datetime')

        elif source == 'yahoo' and period != None:
            self.df = yf.download(symbol, period=period, interval=interval)
            self.df['Datetime'] = self.df.index
            self.df = self.df.set_index('Datetime')
            self.write_file_data(self.df, pure_data, pure_file)

            ind = Indicators(self.df, self.range)
            create_candle_columns(self.df, candle_names=candle_names, candle_rankings=candle_rankings)
            create_candle_label(self.df)
            self.df = self.df.drop(columns=['candlestick_match_count'], axis=1)

            if 'Adj Close' in self.df.columns.to_list():
                self.df = self.df.drop(columns=['Adj Close'], axis=1)
            self.create_binary_feature_label()
            self.df.dropna(inplace= True, how='any')
            self.write_file_data(self.df, path_df, file) 
            
        return self.df
        
    
    def create_binary_feature_label(self) -> None:
        self.df['d_r'] = self.df['Close'].pct_change()
        self.df['temp'] = self.df['d_r']*10000
        self.df['feature_label'] = np.where(self.df['temp'].ge(0), 1, 0)
        self.df['feature_label'] = self.df['feature_label'].shift(-1)
        self.df = self.df.drop(columns=['temp'], axis=1)

    def write_file_data(self, df: pd.DataFrame(), pathdf: str, filedf: str) -> None:
        if not os.path.exists(os.path.join(pathdf, filedf)):
            os.makedirs(pathdf, exist_ok=True)
            with open(os.path.join(pathdf, filedf), mode='a'): pass 
            df.to_csv(os.path.join(pathdf, filedf))
    