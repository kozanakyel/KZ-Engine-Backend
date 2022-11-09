import pandas as pd
import numpy as np
import os
from technical_analysis.indicators import Indicators
import yfinance as yf
from technical_analysis.candlestick_features import *


class DataManipulation():
    def __init__(self, symbol: str, source: str, range_list: list, period=None, interval=None, 
                                start_date=None, end_date=None, scale=1, prefix_path='.'):
        self.symbol = symbol
        self.source = source
        self.scale = scale
        self.range_list = [i*scale for i in range_list]
        self.period = period
        self.interval = interval 
        self.start_date = start_date
        self.end_date = end_date
        self.prefix_path = prefix_path
        self.df = self.create_data_one(self.symbol, self.source, self.period, self.interval, prefix_path=self.prefix_path)

    def get_symbol_df(self, symbol, pure=False):
        path_df = self.prefix_path+'/../data/outputs/data_ind/'+symbol 
        pure_data = self.prefix_path+'/data/pure_data/'+symbol
        pure_file = f'{symbol}_{self.period}_{self.interval}.csv'
        file = f'{symbol}_df_{self.period}_{self.interval}.csv'

        if pure:
            path_df = pure_data
            file = pure_file

        if os.path.exists(os.path.join(path_df, file)):
            df_temp = pd.read_csv(os.path.join(path_df, file))
            df_temp['Datetime'] = pd.to_datetime(df_temp['Datetime'])
            df_temp = df_temp.set_index('Datetime')

        return df_temp
    
    def create_data_one(self, symbol, source, period=None, interval=None, prefix_path='.'):
        path_df = prefix_path+'/data/outputs/data_ind/'+symbol 
        pure_data = prefix_path+'/data/pure_data/'+symbol
        pure_file = f'{symbol}_{period}_{interval}.csv'
        file = f'{symbol}_df_{period}_{interval}.csv'
        
        if os.path.exists(os.path.join(path_df, file)):
            self.df = pd.read_csv(os.path.join(path_df, file))
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
            self.df = self.df.set_index('Datetime')

        elif source == 'yahoo' and period != None:
            df_download = yf.download(symbol, period=period, interval=interval)
            self.df = df_download.copy()
            self.df['Datetime'] = self.df.index
            self.df = self.df.set_index('Datetime')
            if self.df.shape[0] > self.range_list[-1]:
                self.write_file_data(self.df, pure_data, pure_file)
                
                indicators = Indicators(self.df, self.range_list)
                indicators.create_ind_features()
                self.df = indicators.df.copy()
                create_candle_columns(self.df, candle_names=candle_names, candle_rankings=candle_rankings)
                create_candle_label(self.df)
                self.df = self.df.drop(columns=['candlestick_match_count'], axis=1)

                if 'Adj Close' in self.df.columns.to_list():
                    self.df = self.df.rename(columns={"Adj Close": "Adj_close"})
                self.create_binary_feature_label()
                self.df.dropna(inplace= True, how='any')
                self.write_file_data(self.df, path_df, file) 
            else:
                print(f'{self.symbol} shape size not sufficient from download')
                return None
            
        return self.df
        
    
    def create_binary_feature_label(self) -> None:
        self.df['d_r'] = self.df['Close'].pct_change()
        self.df['temp'] = self.df['d_r']*10000
        self.df['feature_label'] = np.where(self.df['temp'].ge(0), 1, 0)
        self.df['feature_label'] = self.df['feature_label'].shift()
        self.df = self.df.drop(columns=['temp'], axis=1)

    def write_file_data(self, df: pd.DataFrame(), pathdf: str, filedf: str) -> None:
        if not os.path.exists(os.path.join(pathdf, filedf)):
            os.makedirs(pathdf, exist_ok=True)
            with open(os.path.join(pathdf, filedf), mode='a'): pass 
            df.to_csv(os.path.join(pathdf, filedf))

    def extract_features(self):
        path_df = self.prefix_path+'/data/outputs/feature_data/'+self.symbol
        file_df = f'{self.symbol}_df_{self.period}_{self.interval}.csv'
        
        if os.path.exists(os.path.join(path_df, file_df)):
            sample = pd.read_csv(os.path.join(path_df, file_df))
            sample['Datetime'] = pd.to_datetime(sample['Datetime'])
            sample = sample.set_index('Datetime')
        else:
            df = self.df.copy()
            sample = self.pattern_helper_for_extract_feature(df)        
            self.norm_features_ind(sample, df, 'ema', self.range_list, df.Close)
            self.norm_features_ind(sample, df, 'mfi', self.range_list, 100)
            self.norm_features_ind(sample, df, 'sma', self.range_list, df.Close)
            self.norm_features_ind(sample, df, 'wma', self.range_list, df.Close)
            self.norm_features_ind(sample, df, 'tema', self.range_list, df.Close)
            self.norm_features_ind(sample, df, 'kama', self.range_list, df.Close)
            self.norm_features_ind(sample, df, 'rsi', self.range_list, 100)
            self.norm_adx_ind(sample, df, self.range_list)
            sample['st_ich'] = (df['ich_tline'] > df['ich_kline']).astype(int)
        
            sample['month'] = sample.index.month
            sample['weekday'] = sample.index.weekday
            sample['is_quarter_end'] = sample.index.is_quarter_end*1
            sample['candle'] = df.candle_label
            sample['vol_delta'] = sample['Volume'].pct_change()
            self.add_lags(sample, df, 5)
            sample['log_return'] = df.log_rt
            sample.drop(columns=['Close', 'Volume'], axis=1, inplace=True)
            self.write_file_data(sample, path_df, file_df)

        return sample

    def norm_features_ind(self, sampledf, df, ind, range_list, dividend):
        k = 0
        for i in range_list:
            sampledf[f'st_{ind}_{i}'] = (dividend - df[f'{ind}_{i}']) / dividend
            if k % 2 == 1:
                sampledf[f'st_cut_{ind}_{range_list[k-1]}_{range_list[k]}'] = \
                        (df[f'{ind}_{range_list[k-1]}'] > df[f'{ind}_{range_list[k]}']).astype(int)
            k += 1

    def norm_adx_ind(self, sampledf, df, range_list):
        for i in range_list:
            sampledf[f'st_adx_{i}'] = (100 - df[f'adx_{i}']) / 100
            pattern1 = df[f'adx_{i}'] < 50
            pattern2 = df[f'dmi_up_{i}'] > df[f'dmi_down_{i}']
            sampledf[f'st_adxdmi_{i}'] = (pattern2).astype(int) + pattern1

    def add_lags(self, sampledf: pd.DataFrame(), df: pd.DataFrame(), lag_numbers: int) -> None:
        i = 1
        while i < lag_numbers:
            sampledf[f'lag_{i}'] = (df.log_rt.shift(i) > 0).astype(int)
            i += 1

    def compound_annual_growth_rate(df: pd.DataFrame(), close) -> float:
        norm = df[close][-1]/df[close][0]
        cagr = (norm)**(1/((df.index[-1] - df.index[0]).days / 365.25)) - 1
        return cagr

    def pattern_helper_for_extract_feature(self, df):
        sample  = df[['Open','High','Low','Close','Volume']].copy()
        sample.drop(columns=['Open','High','Low'], axis=1, inplace=True)
        sample['st_hisse'] = np.nan
        sample['st_stoch'] = np.nan
        sample['st_fisher'] = np.nan
        sample['st_ich'] = np.nan
        sample['st_mfi'] = np.nan

        for index, datetime in enumerate(df.index):
            current_datetime = datetime
            ema5 = df['ema_5'].iloc[index]
            sma10 = df['sma_10'].iloc[index]
            macd = df['macd'].iloc[index]
            macds = df['macdsignal'].iloc[index]
            ichkline = df['ich_kline'].iloc[index]
            ichtline = df['ich_tline'].iloc[index]
            close = df['Close'].iloc[index]
            dmu = df['dmi_up_15'].iloc[index]
            dmd = df['dmi_down_15'].iloc[index]
            stk = df['stoch_k'].iloc[index]
            std = df['stoch_d'].iloc[index]
            fischer = df['FISHERT_9_1'].iloc[index]
            mfi = df['mfi_20'].iloc[index]

            if stk >= std:
                sample['st_stoch'].iloc[index] = 1
            else:
                sample['st_stoch'].iloc[index] = 0

            if fischer >= 2.5:
                sample['st_fisher'].iloc[index] = 3
            elif fischer >= -2.5:
                sample['st_fisher'].iloc[index] = 2
            else:
                sample['st_fisher'].iloc[index] = 1 

            if mfi >= 75:
                sample['st_mfi'].iloc[index] = 3
            elif mfi >= 25:
                sample['st_mfi'].iloc[index] = 2
            else:
                sample['st_mfi'].iloc[index] = 1 

            pattern1 = ema5 >= sma10
            pattern2 = macd > macds
            pattern3 = ichkline < close
            pattern4 = dmu >= dmd
            pattern5 = sma10 < close
            pattern6 = stk > std

            all_pattern = (pattern1 and pattern2 and pattern3 and pattern4 and pattern5 and pattern6)
            if all_pattern:
                sample['st_hisse'].iloc[index] = 1
            else:
                sample['st_hisse'].iloc[index] = 0

        return sample
    