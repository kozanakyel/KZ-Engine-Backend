import pandas as pd
import talib
import pandas_ta as ta
from technical_analysis.candlestick_features import *
"""
@author: Ugur AKYEL
@description: For creating columns and Dataframes with related to technical analysis and indicators
                each talib or pandas_ta library using for any programming environment you can
                work this code.
"""


class Indicators():

    def __init__(self, df: pd.DataFrame(), range_list: list):
        self.range_list = range_list
        self.df = df.copy()

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
        self.create_ind_with_hlct(self.df, 'dmi_up', talib.PLUS_DI, self.range_list)
        self.create_ind_with_hlct(self.df, 'dmi_down', talib.MINUS_DI, self.range_list)
        self.create_ind_with_ct(self.df, 'cmo', talib.CMO, self.range_list)
        self.create_macd(self.df, talib.MACD)
        self.create_ind_with_hlct(self.df, 'cci', talib.CCI, self.range_list)
        self.create_ind_with_ct(self.df, 'rsi', talib.RSI, self.range_list)
        self.create_ind_with_hlct(self.df, 'wllr', talib.WILLR, self.range_list)
        self.create_ind_with_hlcvt(self.df, 'mfi', talib.MFI, self.range_list)
        self.create_ind_cv(self.df, 'obv', talib.OBV)
        self.create_ind_hlcv(self.df, 'ad_cha', talib.AD)
        self.create_ind_with_hlct(self.df, 'atr', talib.ATR, self.range_list)
        self.create_ind_with_hlct(self.df, 'natr', talib.NATR, self.range_list)
        self.create_ind_with_ct(self.df, 'roc', talib.ROC, self.range_list)
        self.create_ind_with_c_stoch(self.df, talib.STOCHRSI)
        self.create_ichmiouk_kijunsen(self.df)
        self.create_ichmiouk_tenkansen(self.df)
        self.df.ta.fisher(append=True)
        create_candle_columns(self.df, candle_names=candle_names, candle_rankings=candle_rankings)
        create_candle_label(self.df)
        self.df = self.df.drop(columns=['candlestick_match_count'], axis=1)
        self.df['log_rt'] = self.df.ta.log_return()

    def create_ind_with_ct(self, dft: pd.DataFrame(), ind: str, func_ta, range_list: list) -> None:
        for i in range_list:
            dft[ind+'_'+str(i)] = func_ta(dft['close'], timeperiod=i)

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

    def create_ind_with_hlct(self, dft: pd.DataFrame(), ind: str, func_ta, range_list: list) -> None:
        for i in range_list:
            dft[ind+'_'+str(i)] = func_ta(dft['high'], dft['low'], dft['close'], timeperiod=i)

    def create_bband_t(self, dft: pd.DataFrame(), func_ta, range_list: list, nbdevup=1, nbdevdn=1, matype=0) -> None:
        for i in range_list:
            dft['upband_'+str(i)], dft['midband_'+str(i)], dft['lowband_'+str(i)] = \
                        func_ta(dft['close'], timeperiod=i, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    def create_macd(self, dft: pd.DataFrame(), func_ta, fastperiod=12, slowperiod=26, signalperiod=9) -> None:
        dft['macd'], dft['macdsignal'], dft['macdhist'] = \
                func_ta(dft['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    
    def create_ind_cv(self, dft: pd.DataFrame(), ind: str, func_ta):
        dft[ind] = func_ta(dft['close'], dft['volume'])

    def create_ind_hlcv(self, dft: pd.DataFrame(), ind: str, func_ta):
        dft[ind] = func_ta(dft['high'], dft['low'], dft['close'], dft['volume'])

    
    def compound_annual_growth_rate(self, df: pd.DataFrame(), close) -> float:
        norm = df[close][-1]/df[close][0]
        cagr = (norm)**(1/((df.index[-1] - df.index[0]).days / 365.25)) - 1
        return cagr

    def create_ind_cols_ta(self) -> None:
        for i in self.range_list:
            self.df.ta.sma(length=i, append=True)