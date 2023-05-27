import pandas as pd
import talib
from tqdm import tqdm
import pandas_ta as ta  # for the fischer indicator only It is necessary

from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.core.domain.base_indicator import BaseIndicator
from KZ_project.ml_pipeline.japanese_candlestick.japanese_candlestick_creator import JapaneseCandlestickCreator


class TalibIndicator(BaseIndicator):
    """
    Represents a class for calculating technical indicators using the TA-Lib library.

    This class provides methods to create various technical indicators based on price and volume data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the price and volume data.
        range_list (list): The list of range values for the indicators.
        logger (Logger): The logger instance for logging messages.

    Methods:
        create_ind_cols(): Creates the indicator columns in the DataFrame.
        create_ind_with_ct(): Creates the indicator columns using the close price and timeperiod.
        create_ind_with_c_stoch(): Creates the Stochastic indicator columns using the close price.
        create_ichmiouk_kijunsen(): Creates the Ichimoku Kijun-sen indicator column.
        create_ichmiouk_tenkansen(): Creates the Ichimoku Tenkan-sen indicator column.
        create_ind_with_hlcvt(): Creates the indicator columns using high, low, close, volume, and timeperiod.
        create_ind_with_hlct(): Creates the indicator columns using high, low, close, and timeperiod.
        create_bband_t(): Creates the Bollinger Bands indicator columns.
        create_macd(): Creates the MACD indicator columns.
        create_ind_cv(): Creates the indicator column using close and volume.
        create_ind_hlcv(): Creates the indicator column using high, low, close, and volume.
        volatility(): Calculates the volatility based on the log returns.

    """

    def __init__(
            self,
            df: pd.DataFrame() = None,
            range_list: list = None,
            logger: Logger = None
    ):
        self.range_list = range_list
        self.logger = logger
        self.df = df.copy()

    def log(self, text):
        """
        Logger implementations for log process
        """
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)

    def create_ind_cols(self) -> None:
        """
        Creates the indicator columns in the DataFrame.

        This method calculates various technical indicators using the TA-Lib library and adds the calculated values as
        columns in the DataFrame.

        Note:
            This method modifies the existing DataFrame in-place.

        Returns:
            None
        """

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

        JapaneseCandlestickCreator.create_candle_columns(self.df)
        JapaneseCandlestickCreator.create_candle_label(self.df)

        self.df = self.df.drop(columns=['candlestick_match_count'], axis=1)
        self.df['daily_return'] = self.df['close'].pct_change()
        self.df['log_return'] = self.df.ta.log_return()

    def create_ind_with_ct(
            self,
            dft: pd.DataFrame,
            ind: str,
            func_ta,
            range_list: list
    ) -> None:
        """
        Creates indicator columns in the DataFrame by applying a TA-Lib function
        that takes the 'close' price and a time period as input.

        Args:
            dft (pd.DataFrame): The DataFrame on which the indicators will be calculated.
            ind (str): The name of the indicator.
            func_ta: The TA-Lib function to calculate the indicator.
            range_list (list): A list of time periods for which the indicator will be calculated.

        Returns:
            None
        """

        for i in tqdm(range_list):
            dft[ind + '_' + str(i)] = func_ta(dft['close'], timeperiod=i)
        self.log(f'Calculated {ind.upper()} for range {range_list}')

    def create_ind_with_c_stoch(self, dft: pd.DataFrame, func_ta) -> None:
        """
        Creates 'stoch_k' and 'stoch_d' columns in the DataFrame by applying a TA-Lib function that calculates the Stochastic RSI.

        Args:
            dft (pd.DataFrame): The DataFrame on which the Stochastic RSI will be calculated.
            func_ta: The TA-Lib function to calculate the Stochastic RSI.

        Returns:
            None
        """
        dft['stoch_k'], dft['stoch_d'] = func_ta(dft['close'], timeperiod=14, fastk_period=3, fastd_period=3,
                                                 fastd_matype=0)

    def create_ichmiouk_kijunsen(self, dft: pd.DataFrame) -> None:
        """
        Creates an 'ich_kline' column in the DataFrame by calculating the Kijun-sen line of the Ichimoku indicator.

        Args:
            dft (pd.DataFrame): The DataFrame on which the Kijun-sen line will be calculated.

        Returns:
            None
        """
        period26_high = dft['high'].rolling(window=26).max()
        period26_low = dft['low'].rolling(window=26).min()
        dft['ich_kline'] = (period26_high + period26_low) / 2

    def create_ichmiouk_tenkansen(self, dft: pd.DataFrame) -> None:
        """
        Creates an 'ich_tline' column in the DataFrame by calculating the Tenkan-sen line of the Ichimoku indicator.

        Args:
            dft (pd.DataFrame): The DataFrame on which the Tenkan-sen line will be calculated.

        Returns:
            None
        """
        period26_high = dft['high'].rolling(window=9).max()
        period26_low = dft['low'].rolling(window=9).min()
        dft['ich_tline'] = (period26_high + period26_low) / 2

    def create_ind_with_hlcvt(
            self,
            dft: pd.DataFrame,
            ind: str,
            func_ta,
            range_list: list
    ) -> None:
        """
        Creates indicator columns in the DataFrame by applying a TA-Lib function
        that takes high, low, close, and volume as inputs.

        Args:
            dft (pd.DataFrame): The DataFrame on which the indicators will be calculated.
            ind (str): The name of the indicator.
            func_ta: The TA-Lib function to calculate the indicator.
            range_list (list): A list of time periods for which the indicator will be calculated.

        Returns:
            None
        """
        for i in range_list:
            dft[ind + '_' + str(i)] = func_ta(dft['high'], dft['low'], dft['close'], dft['volume'], timeperiod=i)
        self.log(f'Calculated {ind.upper()} for range {range_list}')

    def create_ind_with_hlct(
            self,
            dft: pd.DataFrame,
            ind: str,
            func_ta,
            range_list: list
    ) -> None:
        """
        Creates indicator columns in the DataFrame by applying a TA-Lib function that takes high, low, close, and a time period as inputs.

        Args:
            dft (pd.DataFrame): The DataFrame on which the indicators will be calculated.
            ind (str): The name of the indicator.
            func_ta: The TA-Lib function to calculate the indicator.
            range_list (list): A list of time periods for which the indicator will be calculated.

        Returns:
            None
        """
        for i in range_list:
            dft[ind + '_' + str(i)] = func_ta(dft['high'], dft['low'], dft['close'], timeperiod=i)
        self.log(f'Calculated {ind.upper()} for range {range_list}')

    def create_bband_t(
            self,
            dft: pd.DataFrame,
            func_ta,
            range_list: list,
            nbdevup=1,
            nbdevdn=1,
            matype=0
    ) -> None:
        """
        Creates Bollinger Bands columns in the DataFrame by applying a TA-Lib function.

        Args:
            dft (pd.DataFrame): The DataFrame on which the Bollinger Bands will be calculated.
            func_ta: The TA-Lib function to calculate the Bollinger Bands.
            range_list (list): A list of time periods for which the Bollinger Bands will be calculated.
            nbdevup (float, optional): The number of standard deviations to add to the middle band for the upper band. Default is 1.
            nbdevdn (float, optional): The number of standard deviations to subtract from the middle band for the lower band. Default is 1.
            matype (int, optional): The type of moving average to use in the calculation. Default is 0.

        Returns:
            None
        """
        for i in range_list:
            dft['upband_' + str(i)], dft['midband_' + str(i)], dft['lowband_' + str(i)] = \
                func_ta(dft['close'], timeperiod=i, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
        self.log(f'Calculated BOLLINGERBAND for range {range_list}')

    def create_macd(
            self,
            dft: pd.DataFrame,
            func_ta,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
    ) -> None:
        """
        Creates MACD (Moving Average Convergence Divergence) columns in the DataFrame by applying a TA-Lib function.

        Args:
            dft (pd.DataFrame): The DataFrame on which the MACD will be calculated.
            func_ta: The TA-Lib function to calculate the MACD.
            fastperiod (int, optional): The number of periods for the fast EMA. Default is 12.
            slowperiod (int, optional): The number of periods for the slow EMA. Default is 26.
            signalperiod (int, optional): The number of periods for the signal line. Default is 9.

        Returns:
            None
        """
        dft['macd'], dft['macdsignal'], dft['macdhist'] = \
            func_ta(dft['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        self.log(f'Calculated MACD')

    def create_ind_cv(self, dft: pd.DataFrame, ind: str, func_ta):
        """
        Creates an indicator column in the DataFrame by applying a TA-Lib function that takes close and volume as inputs.

        Args:
            dft (pd.DataFrame): The DataFrame on which the indicator will be calculated.
            ind (str): The name of the indicator.
            func_ta: The TA-Lib function to calculate the indicator.

        Returns:
            None
        """
        dft[ind] = func_ta(dft['close'], dft['volume'])
        self.log(f'Calculated {ind.upper()}')

    def create_ind_hlcv(self, dft: pd.DataFrame, ind: str, func_ta):
        """
        Creates an indicator column in the DataFrame by applying a TA-Lib function
        that takes high, low, close, and volume as inputs.

        Args:
            dft (pd.DataFrame): The DataFrame on which the indicator will be calculated.
            ind (str): The name of the indicator.
            func_ta: The TA-Lib function to calculate the indicator.

        Returns:
            None
        """
        dft[ind] = func_ta(dft['high'], dft['low'], dft['close'], dft['volume'])
        self.log(f'Calculated {ind.upper()}')

    def volatility(self, dft: pd.DataFrame):
        """
        Calculates the volatility of the DataFrame based on the log returns.

        Args:
            dft (pd.DataFrame): The DataFrame on which the volatility will be calculated.

        Returns:
            None
        """
        dft['volatility'] = dft['log_return'].std() * 252 ** .5
        self.log(f'Calculated Volatility')
