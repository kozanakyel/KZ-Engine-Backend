import pandas as pd
import talib
import pandas_ta as ta


class Indicators():

    def __init__(self, df: pd.DataFrame(), range: list):
        self.range = range
        self.df = df
        self.create_ind_features()

    def create_ind_features(self) -> None:
        self.create_ind_with_ct('sma', ta.sma, self.range)
        self.create_bband_t(self.range)
        self.create_ind_with_ct('dema', talib.DEMA, self.range)
        self.create_ind_with_ct('ema', talib.EMA, self.range)
        self.create_ind_with_ct('kama', talib.KAMA, self.range)
        self.create_ind_with_ct('t3', talib.T3, self.range)
        self.create_ind_with_ct('tema', talib.TEMA, self.range)
        self.create_ind_with_ct('TRIMA', talib.TRIMA, self.range)
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
        self.create_ind_with_c_stoch(ta.stochrsi)
        self.create_ichmiouk_kijunsen()

    def create_ind_with_ct(self, ind: str, func_ta, range: list) -> None:
        for i in range:
            self.df[ind+'_'+str(i)] = func_ta(self.df['Close'], timeperiod=i)

    def create_ind_with_c_stoch(self, func_ta) -> None:
        self.df['stochrsi_k'], self.df['stochrsi_d'] = func_ta(self.df['Close'])

    def create_ichmiouk_kijunsen(self) -> None:
        period26_high = pd.rolling_max(self.df['High'], window=26)
        period26_low = pd.rolling_min(self.df['Low'], window=26)
        self.df['ich_kline'] = (period26_high + period26_low) / 2

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