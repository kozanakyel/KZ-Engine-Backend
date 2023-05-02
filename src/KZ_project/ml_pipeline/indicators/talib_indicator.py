import pandas as pd
import talib
from tqdm import tqdm

from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.core.strategies.technical_analysis.candlestick_features import *
from KZ_project.ml_pipeline.indicators.base_indicator import BaseIndicator


class TalibIndicator(BaseIndicator):

    def __init__(self, df: pd.DataFrame()=None, range_list: list=None, logger: Logger=None):
        self.range_list = range_list
        self.logger = logger
        self.df = df.copy()

    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)

    def create_ind_candle_cols_talib(self) -> None:
        self.create_ind_with_ct(self.df, 'sma', talib.SMA, self.range_list)
        self.create_bband_t(self.df, talib.BBANDS, self.range_list)
        self.create_ind_with_ct(self.df, 'dema', talib.DEMA, self.range_list)
        self.create_ind_with_ct(self.df, 'ema', talib.EMA, self.range_list)
        self.create_ind_with_ct(self.df, 'kama', talib.KAMA, self.range_list)
        self.create_ind_with_ct(self.df, 't3', talib.T3, self.range_list)
        self.create_ind_with_ct(self.df, 'tema', talib.TEMA, self.range_list)
        self.create_ind_with_ct(self.df, 'trima', talib.TRIMA, self.range_list)
        self.create_ind_with_ct(self.df, 'wma', talib.WMA, self.range_list)
        self.create_ind_with_hlct(self.df, 'adx', talib.ADX, self.range_list)
        self.create_ind_with_hlct(self.df, 'dmn', talib.MINUS_DI, self.range_list)
        self.create_ind_with_hlct(self.df, 'dmp', talib.PLUS_DI, self.range_list)
        self.create_ind_with_ct(self.df, 'cmo', talib.CMO, self.range_list)
        self.create_macd(self.df, talib.MACD)
        self.create_ind_with_hlct(self.df, 'cci', talib.CCI, self.range_list)
        self.create_ind_with_ct(self.df, 'rsi', talib.RSI, self.range_list)
        self.create_ind_with_hlct(self.df, 'willr', talib.WILLR, self.range_list)
        self.create_ind_with_hlcvt(self.df, 'mfi', talib.MFI, self.range_list)
        self.create_ind_cv(self.df, 'obv', talib.OBV)
        self.create_ind_hlcv(self.df, 'ad', talib.AD)
        self.create_ind_with_hlct(self.df, 'atrr', talib.ATR, self.range_list)
        self.create_ind_with_hlct(self.df, 'natr', talib.NATR, self.range_list)
        self.create_ind_with_ct(self.df, 'roc', talib.ROC, self.range_list)
        self.create_ind_with_c_stoch(self.df, talib.STOCHRSI)
        self.create_ichmiouk_kijunsen(self.df)
        self.create_ichmiouk_tenkansen(self.df) 

        fishert = self.df.ta.fisher()
        fishert.columns = ['fishert', 'fisherts']
        self.df = pd.concat([self.df, fishert], axis=1) 

        create_candle_columns(self.df, candle_names=candle_names, candle_rankings=candle_rankings)
        create_candle_label(self.df)

        self.df = self.df.drop(columns=['candlestick_match_count'], axis=1)
        self.df['daily_return'] = self.df['close'].pct_change()
        self.df['log_return'] = self.df.ta.log_return()

    
    def create_ind_with_ct(self, dft: pd.DataFrame(), ind: str, func_ta, range_list: list) -> None:
        for i in tqdm(range_list):
            dft[ind+'_'+str(i)] = func_ta(dft['close'], timeperiod=i)
        self.log(f'Calculated {ind.upper()} for range {range_list}')

    def create_ind_with_c_stoch(self, dft: pd.DataFrame(), func_ta) -> None:
        dft['stoch_k'], dft['stoch_d'] = func_ta(dft['close'], timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)

    def create_ichmiouk_kijunsen(self, dft: pd.DataFrame()) -> None:
        period26_high = dft['high'].rolling(window=26).max()
        period26_low = dft['low'].rolling(window=26).min()
        dft['ich_kline'] = (period26_high + period26_low) / 2

    def create_ichmiouk_tenkansen(self, dft: pd.DataFrame()) -> None:
        period26_high = dft['high'].rolling(window=9).max()
        period26_low = dft['low'].rolling(window=9).min()
        dft['ich_tline'] = (period26_high + period26_low) / 2

    def create_ind_with_hlcvt(self, dft: pd.DataFrame(), ind: str, func_ta, range_list: list) -> None:
        for i in range_list:
            dft[ind+'_'+str(i)] = func_ta(dft['high'], dft['low'], dft['close'], dft['volume'], timeperiod=i) 
        self.log(f'Calculated {ind.upper()} for range {range_list}')

    def create_ind_with_hlct(self, dft: pd.DataFrame(), ind: str, func_ta, range_list: list) -> None:
        for i in range_list:
            dft[ind+'_'+str(i)] = func_ta(dft['high'], dft['low'], dft['close'], timeperiod=i)
        self.log(f'Calculated {ind.upper()} for range {range_list}')

    def create_bband_t(self, dft: pd.DataFrame(), func_ta, range_list: list, nbdevup=1, nbdevdn=1, matype=0) -> None:
        for i in range_list:
            dft['upband_'+str(i)], dft['midband_'+str(i)], dft['lowband_'+str(i)] = \
                        func_ta(dft['close'], timeperiod=i, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
        self.log(f'Calculated BOLLINGERBAND for range {range_list}')

    def create_macd(self, dft: pd.DataFrame(), func_ta, fastperiod=12, slowperiod=26, signalperiod=9) -> None:
        dft['macd'], dft['macdsignal'], dft['macdhist'] = \
                func_ta(dft['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        self.log(f'Calculated MACD')
    
    def create_ind_cv(self, dft: pd.DataFrame(), ind: str, func_ta):
        dft[ind] = func_ta(dft['close'], dft['volume'])
        self.log(f'Calculated {ind.upper()}')

    def create_ind_hlcv(self, dft: pd.DataFrame(), ind: str, func_ta):
        dft[ind] = func_ta(dft['high'], dft['low'], dft['close'], dft['volume'])
        self.log(f'Calculated {ind.upper()}')

    
    def volatility(self, dft: pd.DataFrame()):
        """
        It needs firstly be calculated log_return columns
        """
        dft['volatility'] = dft['log_return'].std()*252**.5
        self.log(f'Calculated Volatility')