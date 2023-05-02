import numpy as np
import pandas as pd
from tqdm import tqdm

from KZ_project.Infrastructure.logger.logger import Logger


class FeatureExtractor():
    
    def __init__(self, df: pd.DataFrame(), range_list: list, interval: str, logger: Logger=None):
        self.logger = logger
        self.range_list = range_list
        self.interval = interval
        self.df = df.copy()
        self.featured_matrix = None
    
    def create_featured_matrix(self):
        self.featured_matrix = self.normalized_all_indicators_col(self.df)
        self.add_datetime_features(self.featured_matrix)
        self.featured_matrix['candle_label'] = self.df.candle_label
        self.featured_matrix['vol_delta'] = (self.featured_matrix['volume'].pct_change() > 0).astype(int)
        self.featured_matrix['log_return'] = self.df.log_return    #### sacma olmus duzelt
        self.add_lags(self.featured_matrix, self.df, 24)
        self.create_binary_feature_label(self.featured_matrix)
        self.log(f'Lags for features and log return vol_delta with binary label')

        self.featured_matrix.drop(columns=['close', 'volume'], axis=1, inplace=True)
        self.featured_matrix = self.featured_matrix.replace([np.inf, -np.inf], np.nan).dropna()
        self.featured_matrix['kz_score'] = self.featured_matrix.sum(axis = 1)/100
        self.log(f'Add KZ score and Index label')
    
    def normalized_all_indicators_col(self, matrix_data):
        sample = self.pattern_helper_for_extract_feature(matrix_data)        
        self.norm_features_ind(sample, matrix_data, 'ema', self.range_list)
        self.norm_features_ind(sample, matrix_data, 'mfi', self.range_list, 100)
        self.norm_features_ind(sample, matrix_data, 'sma', self.range_list)
        self.norm_features_ind(sample, matrix_data, 'wma', self.range_list)
        self.norm_features_ind(sample, matrix_data, 'tema', self.range_list)
        self.norm_features_ind(sample, matrix_data, 'kama', self.range_list)
        self.norm_features_ind(sample, matrix_data, 'rsi', self.range_list, 100)
        self.norm_adx_ind(sample, matrix_data, self.range_list)
        self.log(f'Normalized features for indicators values to 1 and 0')
        return sample
                
        
    def add_datetime_features(self, sample):
        sample['month'] = sample.index.month
        sample['weekday'] = sample.index.weekday
        if self.interval[-1] == 'h':
            sample['hour'] = sample.index.hour
        else:
            sample['hour'] = 0
        sample['is_quarter_end'] = sample.index.is_quarter_end*1
        self.log(f'Add Datetime fetures for extracted feature data')  
    
    
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
    
    def create_binary_feature_label(self, df: pd.DataFrame()) -> None:
        """Next candle Binary feature actual reason forecasting to this label

        Args:
            df (pd.DataFrame):
        """
        df['feature_label'] = (df['log_return'] > 0).astype(int)
        df['feature_label'] = df['feature_label'].shift(-1)
        df["feature_label"][df.index[-1]] = df["feature_label"][df.index[-2]]


    def normalized_df(self, df: pd.DataFrame(), column:str):
        # Normalized process for any specific column or feature
        df[column] = ((df[column] / df[column].mean()) > 1).astype(int)
        
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)