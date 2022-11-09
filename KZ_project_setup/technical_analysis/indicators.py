import pandas as pd
import talib
import pandas_ta as ta
import numpy as np


class Indicators():

    def __init__(self, df: pd.DataFrame(), range_list: list):
        self.range_list = range_list
        self.df = df.copy()

    def create_ind_features(self) -> None:
        self.create_ind_with_ct(self.df, 'sma', talib.SMA, self.range_list)
        self.create_bband_t(self.df, self.range_list)
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
        self.create_macd(self.df)
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
        self.df['log_rt'] = self.df.ta.log_return()

    def create_ind_with_ct(self, dft: pd.DataFrame(), ind: str, func_ta, range_list: list) -> None:
        for i in range_list:
            dft[ind+'_'+str(i)] = func_ta(dft['Close'], timeperiod=i)

    def create_ind_with_c_stoch(self, dft: pd.DataFrame(), func_ta) -> None:
        dft['stoch_k'], dft['stoch_d'] = func_ta(dft['Close'], timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)

    def create_ichmiouk_kijunsen(self, dft: pd.DataFrame()) -> None:
        period26_high = dft['High'].rolling(window=26).max()
        period26_low = dft['Low'].rolling(window=26).min()
        dft['ich_kline'] = (period26_high + period26_low) / 2

    def create_ichmiouk_tenkansen(self, dft: pd.DataFrame()) -> None:
        period26_high = dft['High'].rolling(window=9).max()
        period26_low = dft['Low'].rolling(window=9).min()
        dft['ich_tline'] = (period26_high + period26_low) / 2

    def create_ind_with_hlcvt(self, dft: pd.DataFrame(), ind: str, func_ta, range_list: list) -> None:
        for i in range_list:
            dft[ind+'_'+str(i)] = func_ta(dft['High'], dft['Low'], dft['Close'], dft['Volume'], timeperiod=i) 

    def create_ind_with_hlct(self, dft: pd.DataFrame(), ind: str, func_ta, range_list: list) -> None:
        for i in range_list:
            dft[ind+'_'+str(i)] = func_ta(dft['High'], dft['Low'], dft['Close'], timeperiod=i)

    def create_bband_t(self, dft: pd.DataFrame(), range_list: list, nbdevup=1, nbdevdn=1, matype=0) -> None:
        for i in range_list:
            dft['upband_'+str(i)], dft['midband_'+str(i)], dft['lowband_'+str(i)] = \
                        talib.BBANDS(dft['Close'], timeperiod=i, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    def create_macd(self, dft: pd.DataFrame(), fastperiod=12, slowperiod=26, signalperiod=9) -> None:
        dft['macd'], dft['macdsignal'], dft['macdhist'] = \
                talib.MACD(dft['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    
    def create_ind_cv(self, dft: pd.DataFrame(), ind: str, func_ta):
        dft[ind] = func_ta(dft['Close'], dft['Volume'])

    def create_ind_hlcv(self, dft: pd.DataFrame(), ind: str, func_ta):
        dft[ind] = func_ta(dft['High'], dft['Low'], dft['Close'], dft['Volume'])