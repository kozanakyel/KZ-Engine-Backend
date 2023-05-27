import pandas as pd
from tqdm import tqdm
import pandas_ta as ta

from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.core.domain.base_indicator import BaseIndicator


class PandasTaIndicator(BaseIndicator):
    """
    Custom indicator class that extends the BaseIndicator class and uses the Pandas TA library.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the data.
        range_list (list): List of range values for calculating indicators.
        logger (Logger): Logger object for logging messages.

    Methods:
        create_ind_cols(): Creates the indicator columns in the DataFrame.
        create_ichmiouk_kijunsen(dft): Creates the Ichimoku Kijun-sen indicator column.
        create_ichmiouk_tenkansen(dft): Creates the Ichimoku Tenkan-sen indicator column.
        supertrend(df_temp): Calculates the Supertrend indicator for a temporary DataFrame.
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

    def create_ind_cols(self) -> None:
        """
        Creates the indicator columns in the DataFrame.
        """
        for i in tqdm(self.range_list):
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

    def create_ichmiouk_kijunsen(self, dft: pd.DataFrame()) -> None:
        """
        Creates the Ichimoku Kijun-sen indicator column in the DataFrame.

        Args:
            dft (pd.DataFrame): The DataFrame on which the indicator will be calculated.

        Returns:
            None
        """
        period26_high = dft['high'].rolling(window=26).max()
        period26_low = dft['low'].rolling(window=26).min()
        dft['ich_kline'] = (period26_high + period26_low) / 2

    def create_ichmiouk_tenkansen(self, dft: pd.DataFrame()) -> None:
        """
        Creates the Ichimoku Tenkan-sen indicator column in the DataFrame.

        Args:
            dft (pd.DataFrame): The DataFrame on which the indicator will be calculated.

        Returns:
            None
        """
        period26_high = dft['high'].rolling(window=9).max()
        period26_low = dft['low'].rolling(window=9).min()
        dft['ich_tline'] = (period26_high + period26_low) / 2

    def supertrend(self, df_temp: pd.DataFrame()) -> pd.DataFrame:
        """
        Calculates the Supertrend indicator for a temporary DataFrame.

        Args:
            df_temp (pd.DataFrame): The temporary DataFrame containing the required columns.

        Returns:
            pd.DataFrame: DataFrame with Supertrend indicator columns 'supertrend_support' and 'supertrend_resistance'.
        """
        sti = ta.supertrend(df_temp['high'], df_temp['low'], df_temp['close'], length=7, multiplier=3)
        sti.drop(columns=['SUPERTd_7_3.0', 'SUPERTl_7_3.0'], inplace=True)
        sti.columns = ['supertrend_support', 'supertrend_resistance']
        return sti

    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
