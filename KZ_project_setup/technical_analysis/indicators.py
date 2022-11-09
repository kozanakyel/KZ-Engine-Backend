import pandas as pd
import talib
import pandas_ta as ta
import numpy as np


class Indicators():

    def __init__(self, df: pd.DataFrame(), range: list):
        self.range = range
        self.df = df
        self.create_ind_features()

    def create_ind_features(self) -> None:
        self.create_ind_with_ct('sma', talib.SMA, self.range)
        self.create_bband_t(self.range)
        self.create_ind_with_ct('dema', talib.DEMA, self.range)
        self.create_ind_with_ct('ema', talib.EMA, self.range)
        self.create_ind_with_ct('kama', talib.KAMA, self.range)
        self.create_ind_with_ct('t3', talib.T3, self.range)
        self.create_ind_with_ct('tema', talib.TEMA, self.range)
        self.create_ind_with_ct('trima', talib.TRIMA, self.range)
        self.create_ind_with_ct('wma', talib.WMA, self.range)
        self.create_ind_with_hlct('adx', talib.ADX, self.range)
        self.create_ind_with_hlct('dmi_up', talib.PLUS_DI, self.range)
        self.create_ind_with_hlct('dmi_down', talib.MINUS_DI, self.range)
        self.create_ind_with_ct('cmo', talib.CMO, self.range)
        self.create_macd()
        self.create_ind_with_hlct('cci', talib.CCI, self.range)
        self.create_ind_with_ct('rsi', talib.RSI, self.range)
        self.create_ind_with_hlct('wllr', talib.WILLR, self.range)
        self.create_ind_with_hlcvt('mfi', talib.MFI, self.range)
        self.create_ind_cv('obv', talib.OBV)
        self.create_ind_hlcv('ad_cha', talib.AD)
        self.create_ind_with_hlct('atr', talib.ATR, self.range)
        self.create_ind_with_hlct('natr', talib.NATR, self.range)
        self.create_ind_with_ct('roc', talib.ROC, self.range)
        self.create_ind_with_c_stoch(talib.STOCHRSI)
        self.create_ichmiouk_kijunsen()
        self.create_ichmiouk_tenkansen()
        self.df.ta.fisher(append=True)
        self.df['log_rt'] = self.df.ta.log_return()

    def create_ind_with_ct(self, ind: str, func_ta, range: list) -> None:
        for i in range:
            self.df[ind+'_'+str(i)] = func_ta(self.df['Close'], timeperiod=i)

    def create_ind_with_c_stoch(self, func_ta) -> None:
        self.df['stoch_k'], self.df['stoch_d'] = func_ta(self.df['Close'], timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)

    def create_ichmiouk_kijunsen(self) -> None:
        period26_high = self.df['High'].rolling(window=26).max()
        period26_low = self.df['Low'].rolling(window=26).min()
        self.df['ich_kline'] = (period26_high + period26_low) / 2

    def create_ichmiouk_tenkansen(self) -> None:
        period26_high = self.df['High'].rolling(window=9).max()
        period26_low = self.df['Low'].rolling(window=9).min()
        self.df['ich_tline'] = (period26_high + period26_low) / 2

    def create_ind_with_hlcvt(self, ind: str, func_ta, range: list) -> None:
        for i in range:
            self.df[ind+'_'+str(i)] = func_ta(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], timeperiod=i) 

    def create_ind_with_hlct(self, ind: str, func_ta, range: list) -> None:
        for i in range:
            self.df[ind+'_'+str(i)] = func_ta(self.df['High'], self.df['Low'], self.df['Close'], timeperiod=i)

    def create_bband_t(self, range: list, nbdevup=1, nbdevdn=1, matype=0) -> None:
        for i in range:
            self.df['upband_'+str(i)], self.df['midband_'+str(i)], self.df['lowband_'+str(i)] = \
                        talib.BBANDS(self.df['Close'], timeperiod=i, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    def create_macd(self, fastperiod=12, slowperiod=26, signalperiod=9) -> None:
        self.df['macd'], self.df['macdsignal'], self.df['macdhist'] = \
                talib.MACD(self.df['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    
    def create_ind_cv(self, ind: str, func_ta):
        self.df[ind] = func_ta(self.df['Close'], self.df['Volume'])

    def create_ind_hlcv(self, ind: str, func_ta):
        self.df[ind] = func_ta(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'])

    ## for feature engineering

    def extract_feature(self):
        df = self.df.copy()
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
        
        self.norm_features_ind(sample, df, 'ema', range, df.Close)
        self.norm_features_ind(sample, df, 'mfi', range, 100)
        self.norm_features_ind(sample, df, 'sma', range, df.Close)
        self.norm_features_ind(sample, df, 'wma', range, df.Close)
        self.norm_features_ind(sample, df, 'tema', range, df.Close)
        self.norm_features_ind(sample, df, 'kama', range, df.Close)
        self.norm_features_ind(sample, df, 'rsi', range, 100)
        self.norm_adx_ind(sample, df, range)
        sample['st_ich'] = (df['ich_tline'] > df['ich_kline']).astype(int)

        sample['month'] = sample.index.month
        sample['weekday'] = sample.index.weekday
        sample['is_quarter_end'] = sample.index.is_quarter_end*1
        sample['candle'] = df.candle_label
        sample['vol_delta'] = sample['Volume'].pct_change()
        self.add_lags(sample, df, 5)
        sample['log_return'] = df.log_rt
        sample.drop(columns=['Close', 'Volume'], axis=1, inplace=True)

        return sample

    def norm_features_ind(self, sampledf, df, ind, range, dividend):
        k = 0
        for i in range:
            sampledf[f'st_{ind}_{i}'] = (dividend - df[f'{ind}_{i}']) / dividend
            if k % 2 == 1:
                sampledf[f'st_cut_{ind}_{range[k-1]}_{range[k]}'] = \
                        (df[f'{ind}_{range[k-1]}'] > df[f'{ind}_{range[k]}']).astype(int)
            k += 1

    def norm_adx_ind(self, sampledf, df, range):
        for i in range:
            sampledf[f'st_adx_{i}'] = (100 - df[f'adx_{i}']) / 100
            pattern1 = df[f'adx_{i}'] < 50
            pattern2 = df[f'dmi_up_{i}'] > df[f'dmi_down_{i}']
            sampledf[f'st_adxdmi_{i}'] = (pattern2).astype(int) + pattern1

    def add_lags(self, sampledf: pd.DataFrame(), df: pd.DataFrame(), lag_numbers: int) -> None:
        i = 1
        while i < lag_numbers:
            sampledf[f'lag_{i}'] = (df.log_rt.shift(i) > 0).astype(int)
            i += 1