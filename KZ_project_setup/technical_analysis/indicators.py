import pandas as pd
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

    def create_indicators_columns(self) -> None:
        try:
            import talib
            print('Start TA-LIB module')
            self.create_ind_candle_cols_talib()
            print('created indicators columns with TA-LIB')
        except ModuleNotFoundError:
            print('Start Pandas_ta module')
            self.create_ind_cols_ta()
            print('created indicators columns with Pandas_ta')

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

    def create_ind_cols_ta(self) -> None:
        for i in self.range_list:
            self.df.ta.sma(length=i, append=True)

            bbands = self.df.ta.bbands(length=i).iloc[:, :3]
            bbands.columns = [f'lowband_{i}', f'midband_{i}', f'upband_{i}']
            self.df = pd.concat([self.df, bbands], axis=1)

            self.df.ta.dema(length=i, append=True)
            self.df.ta.ema(length=i, append=True)
            self.df.ta.kama(length=i, append=True)
            self.df = self.df.rename(columns={f'KAMA_{i}_2_30': f'KAMA_{i}'}) 
            self.df.ta.t3(length=i, append=True)
            self.df = self.df.rename(columns={f'T3_{i}_0.7': f'T3_{i}'}) 

            self.df.ta.tema(length=i, append=True)
            self.df.ta.trima(length=i, append=True)
            self.df.ta.wma(length=i, append=True)
            self.df.ta.adx(length=i, append=True)
            self.df.ta.cmo(length=i, append=True)
            self.df.ta.cci(length=i, append=True)
            self.df = self.df.rename(columns={f'CCI_{i}_0.015': f'CCI_{i}'}) 

            self.df.ta.rsi(length=i, append=True)
            self.df.ta.mfi(length=i, append=True)
            self.df.ta.roc(length=i, append=True)
            self.df.ta.willr(length=i, append=True)
            self.df.ta.atr(length=i, append=True)
            self.df = self.df.rename(columns={f'ATRR_{i}': f'ATR_{i}'}) 
            self.df.ta.natr(length=i, append=True)
        
        macd = self.df.ta.macd()
        macd.columns = ['macd', 'macdhist', 'macdsignal']
        self.df = pd.concat([self.df, macd], axis=1)

        stoch = self.df.ta.stochrsi()
        stoch.columns = ['stoch_k', 'stoch_d']
        self.df = pd.concat([self.df, stoch], axis=1)

        self.create_ichmiouk_kijunsen(self.df)
        self.create_ichmiouk_tenkansen(self.df)
        
        fishert = self.df.ta.fisher()
        fishert.columns = ['fishert', 'fisherts']
        self.df = pd.concat([self.df, fishert], axis=1)

        self.df.ta.ad(append=True)
        self.df.ta.obv(append=True)

        self.df['candle_label'] = 0
        self.df['candlestick_pattern'] = 'NO_PATTERN'
        self.df['daily_return'] = self.df['close'].pct_change()
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

