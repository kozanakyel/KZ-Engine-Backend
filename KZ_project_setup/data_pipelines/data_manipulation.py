import pandas as pd
import numpy as np
import os
from technical_analysis.indicators import Indicators
from logger.logger import Logger
import yfinance as yf
import shutil
from tqdm import tqdm
from .file_data_checker import FileDataChecker
from .data_saver import factory_data_saver

"""
@author: Kozan Ugur AKYEL
@since: 07/08/2022

This file main purpose is that getting finance data,
Creating indicators column for Dataframe, 
Cleaning the data, is necessary or if you want
convert the data to featured extracted data with some specific strategies.
Add lags, add, indicators specific strategis and 
obtain nearly binary matrix, 
some strategies need 3 level layer for labelling.
But our main purposes is that matrix preparation for the
our Forecaster model...
"""            
    

class DataManipulation():
    def __init__(self, symbol: str, source: str, range_list: list, period: str=None, interval: str=None, 
                                start_date: str=None, end_date: str=None, scale: int=1, prefix_path: str='.', 
                                main_path: str='/data/outputs/data_ind/', pure_path: str='/data/pure_data/',
                                feature_path: str='/data/outputs/feature_data/', saved_to_csv: bool=True,
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
        
        # DataChecker for file and folder for checking
        # any file or coin info is exist or not
        self.file_data_checker = FileDataChecker(symbol=self.symbol, period=self.period, interval=self.interval, 
                    start_date=self.start_date, end_date=self.end_date, prefix_path=self.prefix_path, 
                    main_path=self.main_path, pure_path=self.pure_path,
                    feature_path=self.feature_path)
        self.saved_to_csv = saved_to_csv
        
        # Factory method for DataSaver CSV ot NO saver
        self.data_saver = factory_data_saver(self.saved_to_csv)
        self.pure_df = None
        self.logger = logger
        if logger != None:
            self.logger = logger
        self.df = self.create_data_with_indicators(self.symbol, self.source, self.period, 
                                        self.interval, self.start_date, 
                                        self.end_date, prefix_path=self.prefix_path)

    def create_data_with_indicators(self, symbol: str, source: str, period: str=None, 
                        interval: str=None, start_date: str=None, 
                        end_date: str=None, prefix_path: str='.'):
        """Created file and checked if file exist or not. Then navigate to data file.
           Download data selcting API Yahoo, Binance, or CTXC..
           Create All the Indicators via the Indicator Class
           Added some features for example JApanese Candlestick label
           from Ta-lib if you have not Talib this features returns the all zeros
           Cleaning the data.... if saved then saved file as .csv

        Args:
            symbol (str): Symbol for asset
            source (str): API source yahoo, binance, ctxc
            period (str, optional): '3m, 30d, 1y, max etc..'. Defaults to None.
            interval (str, optional): 1h, 15m, 1d etc... Defaults to None.
            start_date (str, optional): 2022-12-02 etc.. Defaults to None.
            end_date (str, optional): '2023-02-23 etc... Defaults to None.
            prefix_path (str, optional): '../..' etc.. Defaults to '.'.

        Returns:
            DataFrame: self.df
        """
        #path_df = prefix_path+self.main_path+symbol 
        #pure_data = prefix_path+self.pure_path+symbol

        #if period != None:
        #    pure_file = f'{symbol}_{period}_{interval}.csv'
        #    file = f'{symbol}_df_{period}_{interval}.csv'
        #else:
        #    pure_file = f'{symbol}_{start_date}_{end_date}_{interval}.csv'
        #    file = f'{symbol}_df_{start_date}_{end_date}_{interval}.csv'
        
        #for main indicator file
        if self.file_data_checker.is_main_exist:
            self.df = pd.read_csv(self.file_data_checker.create_main())
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
            self.df = self.df.set_index('Datetime')
            self.log(f'Get data from local file {self.file_data_checker.create_main()}')
            return self.df
        
        elif self.file_data_checker.is_pure_exist:
            self.log('pure file exist')
            df_download = pd.read_csv(self.file_data_checker.create_pure(), index_col=[0], parse_dates=True)

        elif source == 'yahoo':            # Only yahoo download check this data got or not
            df_download = self.yahoo_download(symbol, period=period, interval=interval)
        
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
                self.data_saver.save_data(self.df, self.file_data_checker.pure_folder, 
                                          self.file_data_checker.pure_file)
                self.log(f'Write pure data file to {self.file_data_checker.create_pure()}') 
            indicators = Indicators(self.df, self.range_list, logger=self.logger)    # Create Indicator class fro calculating tech. indicators
            indicators.create_indicators_columns()        
            self.df = indicators.df.copy()
            self.log(f'Created Indicator object and indicators columns')
            
            self.df.columns = self.df.columns.str.lower()              # Columns letters to LOWER
            self.df = self.df.reindex(sorted(self.df.columns), axis=1) # sort Columns with to alphabetic order
            self.create_binary_feature_label(self.df)                  # Add next Candle binary feature
            self.df.dropna(inplace= True, how='any')
            self.log(f'Created Feature label for next day and dropn NaN values')
                
            if self.saved_to_csv:
                self.data_saver.save_data(self.df, self.file_data_checker.main_folder, 
                                          self.file_data_checker.main_file) 
                self.log(f'Write indicator and featured data file to {self.file_data_checker.create_main()}')
        else:
            self.log(f'{self.symbol} shape size not sufficient from download')
            return None
            
        return self.df

    def yahoo_download(self, symbol, period: str=None, interval: str=None, start:str=None, end:str=None) -> pd.DataFrame():
        """For Yahoo API, get data with period interval and specific date.
        you can check wheter yahoo api for getting finance data.

        Returns:
            DataFrame
        """
        if period != None:
            download_df = yf.download(symbol, period=period, interval=interval)
            self.log(f'Get {symbol} data from yahoo period: {period} and interval: {interval}')
        elif start != None:
            download_df = yf.download(symbol, start=start, end=end, interval=interval)
            self.log(f'Get {symbol} data from yahoo start date: {start} and {end} also interval: {interval}')
        else:
            self.log(f'Inappropriate period, interval, start or end, Check please!')
        return download_df
        
    def create_binary_feature_label(self, df: pd.DataFrame()) -> None:
        """Next candle Binary feature actual reason forecasting to this label

        Args:
            df (pd.DataFrame):
        """
        df['feature_label'] = (df['log_return'] > 0).astype(int)
        df['feature_label'] = df['feature_label'].shift(-1)

    def write_file_data(self, df: pd.DataFrame(), pathdf: str, filedf: str) -> None:
        if not os.path.exists(os.path.join(pathdf, filedf)):
            os.makedirs(pathdf, exist_ok=True)
            with open(os.path.join(pathdf, filedf), mode='a'): pass 
            df.to_csv(os.path.join(pathdf, filedf))

    def extract_features(self) -> pd.DataFrame():
        """Create strategies with features and convert it into a 0, 1 range or
           1,2,3 level, for example mfi and adx goes to 1,2,3 also some strategies like
           rsi and others goes to 1 and 0 for 30  level is buy signal and 70 level is sell signal
           so we can seperate 1 and 0 for our mdoel.  

        Returns:
            sample: Dataframe
        """
        path_df = self.prefix_path+self.feature_path+self.symbol
        
        if self.period != None:
            file_df = f'{self.symbol}_df_{self.period}_{self.interval}.csv'
        else:
            file_df = f'{self.symbol}_df_{self.start_date}_{self.end_date}_{self.interval}.csv'
        
        if os.path.exists(os.path.join(path_df, file_df)):
            sample = pd.read_csv(os.path.join(path_df, file_df))
            sample['Datetime'] = pd.to_datetime(sample['Datetime'])
            sample = sample.set_index('Datetime')
            self.log(f'Get features matrix from {path_df} {file_df}')
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
            sample['log_return'] = df.log_return    #### sacma olmus duzelt
            self.add_lags(sample, df, 24)
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
        """Calculate crossover strategies to convert 1 and 0 for featured data matrix
           for all average indicators. sma, ema, wma, t3 etc...

        Args:
            sampledf (dataframe)
            df (dataframe)
            ind (str): indicator name 
            range_list (list)
            dividend (_type_, optional): for normalizations
        """
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
        """Calculate ADX and DMI startegies for deatured data convert 1 and 0
           Because DMI and DMP is important indicators for volatility and volume 
        Args:
            sampledf (_type_)
            df (_type_)
            range_list (_type_)
        """
        for i in range_list:
            sampledf[f'st_adx_{i}'] = (100 - df[f'adx_{i}']) / 100
            pattern1 = df[f'adx_{i}'] < 50
            pattern2 = df[f'dmp_{i}'] > df[f'dmn_{i}']
            sampledf[f'st_adxdmi_{i}'] = (pattern2).astype(int) + pattern1
        self.log(f'Add ADX indicator normalized label')

    def add_lags(self, sampledf: pd.DataFrame(), df: pd.DataFrame(), lag_numbers: int) -> None:
        """Add shifted days with list if you use + sign that means days runs the backs
           but if you use - sign you can represent the future genaration for lags

        Args:
            sampledf (pd.DataFrame): extracted featured data
            df (pd.DataFrame)
            lag_numbers (int): how many days with increasing +1
        """
        i = 2
        while i < lag_numbers:
            sampledf[f'lag_{i}'] = (df.log_return.shift(i) > 0).astype(int)
            i += 1

    def pattern_helper_for_extract_feature(self, df) -> pd.DataFrame():
        """This method convert our strategy for hisses to binary and strategic values for modelling.
         strategies include stoch_rsi, ichnmouku, ema5_ema10 crossover, DMI etc. and all this stuff
         converting to binary matrix. But MFI and FISHERT indicators divided into 3 level like 1,2,3
         Because we can think that two indicators like prefix for the prices...

        Args:
            df (DataFrame): Indicator data matrix

        Returns:
            df (DataFrame)
        """

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
            """Divide 3 level for specific indicators 

            Args:
                x (pd column): float
                params (list): bounded for indiocator values

            Returns:
                int: level 
            """
            if x >= params[0]: 
                return 3
            elif x >= params[1]:
                return 2
            else:
                return 1

        sample['st_mfi'] = df["mfi_15"].apply(lambda x: helper_divide_three(x, params=[75, 25]))
        sample['st_fishert'] = df["fishert"].apply(lambda x: helper_divide_three(x, params=[2.5, -2.5]))
        self.log(f'Hisse strategy, mfi and fischer scoring is finished')

        return sample


    def normalized_df(self, df: pd.DataFrame(), column:str):
        # Normalized process for any specific column or feature
        df[column] = ((df[column] / df[column].mean()) > 1).astype(int)


    def remove_directory(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
            self.log(f'The path is REMOVED: {path}')
        else:
            self.log(f'The path is not exist. {path}')
    
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
            
            
if __name__ == '__main__':
    MAIN_PATH = '/data/outputs/data_ind/'
    PURE_PATH = '/data/pure_data/'
    FEATURE_PATH = '/data/outputs/feature_data/'
    PREFIX_PATH = './KZ_project_setup'
    SYMBOL = 'BTC-USD' 
    PERIOD = None
    INTERVAL = '1h'
    START_DATE = '2020-06-30'
    END_DATE = '2022-07-01'
    
    d = DataManipulation(symbol=SYMBOL, period=PERIOD, interval=INTERVAL, 
                    start_date=START_DATE, end_date=END_DATE, prefix_path=PREFIX_PATH, 
                    main_path=MAIN_PATH, pure_path=PURE_PATH,
                    feature_path=FEATURE_PATH)  
    
    print(os.getcwd())