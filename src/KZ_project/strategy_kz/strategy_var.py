import datetime
from pykalman import KalmanFilter
import warnings
import pandas as pd
import numpy as np
from KZ_project.data_pipelines.data_manipulation import DataManipulation

class StrategyVar():

    def __init__(self, name=None):
        self.name = name

    def create_ema_labels(self, df: pd.DataFrame()):
        df_temp = df.loc['2022-05-01':, :].copy()
        for i in range(1,4):
            df_temp[f'pct_{i}'] = -1 * df_temp.close.pct_change(-i).round(3)
        df_temp['ema12_cross_ema50'] = (df_temp.ema_12 > df_temp.ema_50).astype(int)
        df_temp['ema5_cross_ema10'] = (df_temp.ema_5 > df_temp.ema_10).astype(int)
        df_temp['ema5_cross_ema20'] = (df_temp.ema_5 > df_temp.ema_20).astype(int)
        df_temp['ema9_cross_ema26'] = (df_temp.ema_9 > df_temp.ema_26).astype(int)
        df_temp['ema20_cross_ema50'] = (df_temp.ema_20 > df_temp.ema_50).astype(int)
        df_temp['ema50_cross_ema200'] = (df_temp.ema_50 > df_temp.ema_200).astype(int)
        df_temp['ema12_cross_ema26'] = (df_temp.ema_12 > df_temp.ema_26).astype(int)

        df_result = df_temp[['close', 'volume', 'ema5_cross_ema10', 'ema5_cross_ema20', 'ema9_cross_ema26', 'ema20_cross_ema50',
                                'ema12_cross_ema50', 'ema50_cross_ema200', 'ema12_cross_ema26', 'pct_1', 'pct_2', 'pct_3']].copy()
        df_result.close = df_result.close.round(3)
        df_result.dropna(inplace=True)
        df_result.reset_index(inplace=True)    
        return df_result

    def find_best_cross_strategy(self, symbol: str, df_original: pd.DataFrame()):
        all_strategies = ['ema5_cross_ema10', 'ema5_cross_ema20', 'ema9_cross_ema26', 'ema20_cross_ema50',
                                'ema12_cross_ema50', 'ema50_cross_ema200', 'ema12_cross_ema26']
        result_dict = {}
        max_pct = 0
        df = df_original.copy()
        for k in all_strategies:
            strategy_col = k
            #print(f'\n######### {symbol} - {strategy_col.upper()}  ##########')
            #print(f'Date--------1 gunluk degisim----2 gunluk degisim----3 gunluk degisim')
            df[f'{strategy_col}_shift'] = df[strategy_col].shift()
    
            for i, row in enumerate(df.itertuples()):             
                current = getattr(row, strategy_col)
                prev = getattr(row, f'{strategy_col}_shift')      
                if prev == 0 and current == 1:
                    row_max = max([row.pct_1, row.pct_2, row.pct_3])
                    if row_max > max_pct:
                        max_pct = row_max
                        result_dict[symbol] = (k, max_pct)
                        print(result_dict)
                    #print(f'{row.Datetime.date()}{row.pct_1:>10}{row.pct_2:>20}{row.pct_3:>20}')
        return result_dict

    def kalman_strategy_filter(self, df1: pd.DataFrame(), symbol: str):
        
        df_sample1 = df1.copy()
        df_sample1 = df_sample1[['open', 'high', 'low', 'close', 'volume']]
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize(None)
        df_sample1.index = pd.to_datetime(df_sample1.index, unit='ms')
    
        close = df_sample1['close'][len(df_sample1)-1]
        low = df_sample1['low'][len(df_sample1)-1]
        # Construct a Kalman filter
        kf = KalmanFilter(transition_matrices = [1],    # The value for At. It is a random walk so is set to 1.0
                      observation_matrices = [1],   # The value for Ht.
                      initial_state_mean = 0,       # Any initial value. It will converge to the true state value.
                      initial_state_covariance = 1, # Sigma value for the Qt in Equation (1) the Gaussian distribution
                      observation_covariance=1,     # Sigma value for the Rt in Equation (2) the Gaussian distribution
                      transition_covariance=.01)    # A small turbulence in the random walk parameter 1.0
        # Get the Kalman smoothing
        state_means, _ = kf.filter(df_sample1['close'].values)
        # Call it kf_mean
        df_sample1['kf_mean'] = np.array(state_means)
        kalman = df_sample1.kf_mean[len(df_sample1)-1]
        aboveKalman = low > kalman
        # exponential moving averages 
        ema_14 = df_sample1.ta.ema(14, append=True)[-1:].reset_index(drop=True)[0]
        ema_91 = df_sample1.ta.ema(91, append=True)[-1:].reset_index(drop=True)[0]
        ema_crossover = ema_14 > ema_91    
        # lower/upper 14-day bollinger bands for mean reversion
        bbl_14 = df_sample1.ta.bbands(length=14, append=True)[['BBL_14_2.0']].tail(1).values[0][0]
        bbu_14 = df_sample1.ta.bbands(length=14, append=True)[['BBU_14_2.0']].tail(1).values[0][0]
        bband_buy = close < bbl_14
        bband_sell = close > bbu_14
        # ichimoku 9 & 26-day forecasts 
        # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.IchimokuIndicator
        isa_9 = df_sample1.ta.ichimoku()[1]['ISA_9'].tail(1).values[0] # help(ta.ichimoku)
        isb_26 = df_sample1.ta.ichimoku()[1]['ISB_26'].tail(1).values[0]
        # archer ma 
        # https://github.com/twopirllc/pandas-ta#general
        amat = (df_sample1.ta.amat()['AMATe_LR_8_21_2'].tail(1).values[0] == 1)
        # rsi
        rsi = df_sample1.ta.rsi()[len(df_sample1)-1]
        rsi_buy = rsi < 30
        rsi_sell = rsi > 70
        # choppy
        # https://github.com/twopirllc/pandas-ta#trend-18
        try: 
            chop = "{:.2f}".format(df_sample1.ta.chop()[len(df_sample1.ta.chop())-1]) 
        except warnings.RunTimeWarning:
            chop = 0
        # signal
        buy = (close < isa_9) & (close < isb_26) & amat & rsi_buy & bband_buy & aboveKalman
        sell = (close > isa_9) & (close > isb_26) & ~amat & rsi_sell & bband_sell & ~aboveKalman

        return df_sample1, symbol, close, isa_9, isb_26, chop, rsi, amat, ema_crossover, buy, sell, aboveKalman

    def get_kalman_filter_result(self, data: DataManipulation, symbol_list: list, 
                            write_path_file='./results_kalman.csv', to_csv=True) -> pd.DataFrame():

        results = []
        symbols = []
        today = pd.Timestamp(datetime.datetime.now()).strftime("%Y-%m-%d")

        # iteration
        for symbol in symbol_list:
            try:
                output = self.kalman_strategy_filter(data, symbol)
                results.append({'Symbol':output[1],
                        'Long':output[9],
                        'Short':output[10],
                        'Date':today,
                        'Close':output[2],
                        'Ichimoku 9-Day Forecast':output[3],
                        'Ichimoku 26-Day Forecast':output[4],
                        'Choppiness (%)':output[5],
                        'RSI':output[6],
                        'Archer MA Trending':output[7],
                        'EMA14 > EMA91':output[8],
                        'Low > Kalman':output[11],
                    })
                symbols.append(symbol)
            except:
                pass
        results = pd.DataFrame(results)
        if to_csv:
            results.to_csv(write_path_file)

        return results


    

