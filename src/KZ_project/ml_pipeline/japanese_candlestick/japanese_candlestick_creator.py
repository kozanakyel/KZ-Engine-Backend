
from itertools import compress
import numpy as np
import pandas as pd
import talib

from KZ_project.ml_pipeline.japanese_candlestick.const_japanese_candlestick import CANDLE_NAMES, CANDLE_RANKINGS

class JapaneseCandlestickCreator():
    
    @staticmethod
    def create_candle_columns(df: pd.DataFrame(), candle_names=CANDLE_NAMES, candle_rankings=CANDLE_RANKINGS) -> None:
        open=df.open 
        high=df.high 
        low=df.low 
        close=df.close
        # create columns for each pattern
        for candle in candle_names:
            df[candle] = getattr(talib, candle)(open, high, low, close)

        df['candlestick_match_count'] = np.nan
        df['candlestick_pattern'] = np.nan

        for index, row in df.iterrows():
            # no pattern found
            if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
                df.loc[index, 'candlestick_match_count'] = 0
                df.loc[index,'candlestick_pattern'] = "NO_PATTERN"        
                # single pattern found
            elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
                    # bull pattern 100 or 200
                if any(row[candle_names].values > 0):
                    pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] = 1
                # bear pattern -100 or -200
                else:
                    pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] = 1
                # multiple patterns matched -- select best performance
            else:
                # filter out pattern names from bool list of values
                patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
                container = []
                for pattern in patterns:
                    if row[pattern] > 0:
                        container.append(pattern + '_Bull')
                    else:
                        container.append(pattern + '_Bear')
                rank_list = [candle_rankings[p] for p in container]
                if len(rank_list) == len(container):
                    rank_index_best = rank_list.index(min(rank_list))
                    df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                    df.loc[index, 'candlestick_match_count'] = len(container)
            # clean up candle columns
        df.drop(candle_names, axis = 1, inplace = True)

    @staticmethod
    def create_candle_label(df: pd.DataFrame()) -> None:
        df['candle_label'] = np.nan
        for i, datetime in enumerate(df.index):
            if df.loc[datetime, 'candlestick_pattern'].endswith('Bull'):
                df.loc[datetime, 'candle_label'] = 1
            elif df.loc[datetime, 'candlestick_pattern'].endswith('Bear'):
                df.loc[datetime, 'candle_label'] = -1
            else:
                df.loc[datetime, 'candle_label'] = 0
        df = df.drop(columns=['candlestick_match_count'], axis=1)