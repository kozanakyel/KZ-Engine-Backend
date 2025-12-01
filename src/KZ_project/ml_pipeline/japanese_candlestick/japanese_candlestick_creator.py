import numpy as np
import pandas as pd

from KZ_project.ml_pipeline.japanese_candlestick.const_japanese_candlestick import CANDLE_NAMES, CANDLE_RANKINGS


class JapaneseCandlestickCreator:
    @staticmethod
    def create_candle_columns(df: pd.DataFrame(), candle_names=CANDLE_NAMES, candle_rankings=CANDLE_RANKINGS) -> None:
        """Populate placeholder candlestick columns.

        TA-LIB-driven candlestick detection has been removed. This helper now fills
        neutral placeholder values so downstream feature builders can continue to
        run without external TA dependencies.
        """
        df["candlestick_match_count"] = 0
        df["candlestick_pattern"] = "NO_PATTERN"

    @staticmethod
    def create_candle_label(df: pd.DataFrame()) -> None:
        df["candle_label"] = 2
        df.drop(columns=["candlestick_match_count"], axis=1, inplace=True)
