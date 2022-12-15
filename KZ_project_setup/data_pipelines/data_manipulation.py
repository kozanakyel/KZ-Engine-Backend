import pandas as pd
import numpy as np
import os
from technical_analysis.indicators import Indicators
from logger.logger import Logger
import yfinance as yf
import shutil
from tqdm import tqdm

class DataManipulation():
    def __init__(self, symbol: str, source: str, range_list: list, period=None, interval=None, 
                                start_date=None, end_date=None, scale=1, prefix_path='.', 
                                main_path='/data/outputs/data_ind/', pure_path='/data/pure_data/',
                                feature_path='/data/outputs/feature_data/', saved_to_csv=True,
                                logger: Logger=None):
        self.symbol = symbol
        self.source = source
        self.scale = scale
        self.range_list = [i*scale for i in range_list]
        self.period = period
        self.interval = interval 
        self.start_date = start_date
        self.end_date = end_date
        self.prefix_path = prefix_path
        self.main_path = main_path
        self.pure_path = pure_path
        self.feature_path = feature_path
        self.saved_to_csv = saved_to_csv
        self.pure_df = None
        if logger != None:
            self.logger = logger
        self.df = self.create_data_one(self.symbol, self.source, self.period, 
                                        self.interval, self.start_date, 
                                        self.end_date, prefix_path=self.prefix_path)
        

    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)

    def create_data_one(self, symbol, source, period=None, 
                        interval=None, start_date=None, 
                        end_date=None, prefix_path='.'):

        path_df = prefix_path+self.main_path+symbol 
        pure_data = prefix_path+self.pure_path+symbol

        if period != None:
            pure_file = f'{symbol}_{period}_{interval}.csv'
            file = f'{symbol}_df_{period}_{interval}.csv'
        else:
            pure_file = f'{symbol}_{start_date}_{end_date}_{interval}.csv'
            file = f'{symbol}_df_{start_date}_{end_date}_{interval}.csv'
        
        if os.path.exists(os.path.join(path_df, file)):
            self.df = pd.read_csv(os.path.join(path_df, file))
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
            self.df = self.df.set_index('Datetime')
            self.log(f'Get data from local file {os.path.join(path_df, file)}')
            return self.df
        
        elif os.path.exists(os.path.join(pure_data, pure_file)):
            self.log('pure file exist')
            df_download = pd.read_csv(os.path.join(pure_data, pure_file), index_col=True, parse_dates=True)

        elif source == 'yahoo' and period != None:
            df_download = yf.download(symbol, period=period, interval=interval)
            self.log(f'Get {symbol} data from yahoo period: {period} and interval: {interval}')
        else:
            df_download = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            self.log(f'Get {symbol} data from yahoo start date: {start_date} and {end_date} also interval: {interval}')

        self.df = df_download.copy()
        self.df['Datetime'] = self.df.index
        self.df = self.df.set_index('Datetime')
        self.df.dropna(inplace=True)

        if self.df.shape[0] > self.range_list[-1]:
            self.df.columns = self.df.columns.str.lower()
            if 'adj close' in self.df.columns.to_list():
                self.df = self.df.rename(columns={'adj close': 'adj_close'})  
            self.pure_df = self.df.copy() 
            if self.saved_to_csv:
                self.write_file_data(self.df, pure_data, pure_file)
                self.log(f'Write pure data file to {pure_data+pure_file}')
                
            indicators = Indicators(self.df, self.range_list)
            indicators.create_indicators_columns()
            self.df = indicators.df.copy()
            self.log(f'Created Indicator object and indicators columns')
            
            self.df.columns = self.df.columns.str.lower()
            self.df = self.df.reindex(sorted(self.df.columns), axis=1) 
            self.create_binary_feature_label(self.df)
            self.df.dropna(inplace= True, how='any')
            self.log(f'Created Feature label for next day and dropn NaN values')
                
            if self.saved_to_csv:
                self.write_file_data(self.df, path_df, file) 
                self.log(f'Write indicator and featured data file to {path_df+file}')
        else:
            self.log(f'{self.symbol} shape size not sufficient from download')
            return None
            
        return self.df
        
    def create_binary_feature_label(self, df: pd.DataFrame()) -> None:
        df['feature_label'] = (df['log_return'] > 0).astype(int)
        df['feature_label'] = df['feature_label'].shift(-1)

    def write_file_data(self, df: pd.DataFrame(), pathdf: str, filedf: str) -> None:
        if not os.path.exists(os.path.join(pathdf, filedf)):
            os.makedirs(pathdf, exist_ok=True)
            with open(os.path.join(pathdf, filedf), mode='a'): pass 
            df.to_csv(os.path.join(pathdf, filedf))

    def extract_features(self) -> pd.DataFrame():
        path_df = self.prefix_path+self.feature_path+self.symbol
        
        if self.period != None:
            file_df = f'{self.symbol}_df_{self.period}_{self.interval}.csv'
        else:
            file_df = f'{self.symbol}_df_{self.start_date}_{self.end_date}_{self.interval}.csv'
        
        if os.path.exists(os.path.join(path_df, file_df)):
            sample = pd.read_csv(os.path.join(path_df, file_df))
            sample['Datetime'] = pd.to_datetime(sample['Datetime'])
            sample = sample.set_index('Datetime')
        else:
            df = self.df.copy()
            sample = self.pattern_helper_for_extract_feature(df)        
            self.norm_features_ind(sample, df, 'ema', self.range_list)
            self.norm_features_ind(sample, df, 'mfi', self.range_list, 100)
            self.norm_features_ind(sample, df, 'sma', self.range_list)
            self.norm_features_ind(sample, df, 'wma', self.range_list)
            self.norm_features_ind(sample, df, 'tema', self.range_list)
            self.norm_features_ind(sample, df, 'kama', self.range_list)
            self.norm_features_ind(sample, df, 'rsi', self.range_list, 100)
            self.norm_adx_ind(sample, df, self.range_list)
            self.log(f'Normalized features for indicators values to 1 and 0')
        
            # date normalization and find date weightys for daily return!!!!!!
            sample['month'] = sample.index.month
            sample['weekday'] = sample.index.weekday
            if self.interval[-1] == 'h':
                sample['hour'] = sample.index.hour
            else:
                sample['hour'] = 0
            sample['is_quarter_end'] = sample.index.is_quarter_end*1
            self.log(f'Add Datetime fetures for extracted feature data')

            sample['candle_label'] = df.candle_label
            sample['vol_delta'] = (sample['volume'].pct_change() > 0).astype(int)
            sample['log_return'] = df.log_return
            self.add_lags(sample, df, 9)
            self.create_binary_feature_label(sample)
            self.log(f'Lags for features and log return vol_delta with binary label')

            sample.drop(columns=['close', 'volume'], axis=1, inplace=True)
            sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
            sample['kz_score'] = sample.sum(axis = 1)/100
            self.log(f'Add KZ score and Index label')
            if self.saved_to_csv:
                self.write_file_data(sample, path_df, file_df)

        return sample

    def norm_features_ind(self, sampledf, df, ind, range_list, dividend=None) -> None:
        k = 0
        self.log(f'Start {ind} normalized label')
        for i in tqdm(range_list):
            if dividend != None:
                sampledf[f'st_{ind}_{i}'] = df[f'{ind}_{i}'] / dividend
            else:
                sampledf[f'st_{ind}_{i}'] = (df[f'{ind}_{i}'].pct_change() > 0).astype(int)
                if k % 2 == 1:
                    sampledf[f'st_cut_{ind}_{range_list[k-1]}_{range_list[k]}'] = \
                            (df[f'{ind}_{range_list[k-1]}'] > df[f'{ind}_{range_list[k]}']).astype(int)
            k += 1
        self.log(f'Add {ind} normalized label')

    def norm_adx_ind(self, sampledf, df, range_list) -> None:
        for i in range_list:
            sampledf[f'st_adx_{i}'] = (100 - df[f'adx_{i}']) / 100
            pattern1 = df[f'adx_{i}'] < 50
            pattern2 = df[f'dmp_{i}'] > df[f'dmn_{i}']
            sampledf[f'st_adxdmi_{i}'] = (pattern2).astype(int) + pattern1
        self.log(f'Add ADX indicator normalized label')

    def add_lags(self, sampledf: pd.DataFrame(), df: pd.DataFrame(), lag_numbers: int) -> None:
        i = 2
        while i < lag_numbers:
            sampledf[f'lag_{i}'] = (df.log_return.shift(i) > 0).astype(int)
            i += 1

    def pattern_helper_for_extract_feature(self, df) -> pd.DataFrame():

        self.log(f'Start helper function for Hisse strategy')
        sample  = df[['open','high','low','close','volume']].copy()
        sample.drop(columns=['open','high','low'], axis=1, inplace=True)
        
        sample['st_stoch'] = (df['stoch_k'] > df['stoch_d']).astype(int)
        sample['st_ich'] = (df['ich_tline'] > df['ich_kline']).astype(int)  
        sample['st_cut_ema5_sma10'] = (df['ema_5'] >= df['sma_10']).astype(int)
        sample['st_macd'] = (df['macd'] >= df['macdsignal']).astype(int)
        sample['st_ich_close'] = (df['ich_kline'] < df['close']).astype(int)
        sample['st_dmi'] = (df['dmp_15'] > df['dmn_15']).astype(int)
        sample['st_cut_sma10_close'] = (df['sma_10'] > df['close']).astype(int)

        sample['st_hisse'] = sample['st_stoch'] & sample['st_ich'] & sample['st_cut_ema5_sma10'] \
                    & sample['st_macd'] & sample['st_ich_close'] & sample['st_dmi'] & sample['st_cut_sma10_close']

        def helper_divide_three(x, params: list):
            if x >= params[0]: 
                return 3
            elif x >= params[1]:
                return 2
            else:
                return 1

        sample['st_mfi'] = df["mfi_15"].apply(lambda x: helper_divide_three(x, params=[75, 25]))
        sample['st_fishert'] = df["fishert"].apply(lambda x: helper_divide_three(x, params=[2.5, -2.5]))
        self.log(f'SHisse strategy, mfi and fischer scoring is finished')

        return sample


    def normalized_df(self, df: pd.DataFrame(), column:str):
        df[column] = ((df[column] / df[column].mean()) > 1).astype(int)


    def remove_directory(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
            self.log(f'The path is REMOVED: {path}')
        else:
            self.log(f'The path is not exist. {path}')
    