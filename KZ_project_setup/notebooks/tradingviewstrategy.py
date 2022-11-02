import pandas as pd
from tradingview_ta import TA_Handler, Interval, Exchange, Analysis
from tradingview_ta import *
import time


class TradingViewStrategy():
    def __init__(self, exchange, screener, path, index='Order'):
        self.exchange = exchange
        self.screener = screener
        self.path = path
        self.index = index
        self.symbols = self.get_symbols(self.path, self.index)
        self.symbols_dict = self.get_symbols_dict()
        self.intervals = self.get_intervals()


    def get_symbols(self, path, index) -> list:
        df = pd.read_csv(path, index_col=index)
        df_symbols = df['Code'].to_list()
        df_symbols = [f'{self.exchange}:{i.lower()}' for i in df_symbols]
        return df_symbols

    def get_symbols_dict(self) -> dict:
        df_symbols_dict = [(i.upper(), -1) for i in self.symbols] 
        df_symbols_dict = dict(df_symbols_dict)
        return df_symbols_dict

    def get_intervals(self) -> dict:
        intervals = {'1m': Interval.INTERVAL_1_MINUTE,
                     '5m': Interval.INTERVAL_5_MINUTES,
                     '15m': Interval.INTERVAL_15_MINUTES,
                     '30m': Interval.INTERVAL_30_MINUTES,
                     '1h': Interval.INTERVAL_1_HOUR,
                     '2h': Interval.INTERVAL_2_HOURS,
                     '4h': Interval.INTERVAL_4_HOURS,
                     '1d': Interval.INTERVAL_1_DAY,
                     '1w': Interval.INTERVAL_1_WEEK,
                     '1mo': Interval.INTERVAL_1_MONTH}
        return intervals

    def get_one_analysis(self, symbol, interval) -> Analysis:
        symbol_analysis = TA_Handler(
            symbol=symbol,
            screener=self.screener,
            exchange=self.exchange,
            interval=self.intervals[interval]
        )
        return symbol_analysis.get_analysis()

    def ind_st_buy_status(self, stock_ind: dict, symbol: str):
    
        current_price = self.get_one_analysis(symbol.split(':')[1], '1m')
        current_price = current_price.indicators['close']
    
        pattern1 = stock_ind['EMA5'] >= stock_ind['SMA10']
        pattern2 = stock_ind['MACD.macd'] > stock_ind['MACD.signal']
        pattern3 = stock_ind['Ichimoku.BLine'] < current_price
        pattern4 = stock_ind['ADX+DI'] >= stock_ind['ADX-DI']
        pattern5 = stock_ind['SMA10'] < current_price
        pattern6 = stock_ind['Stoch.K'] > stock_ind['Stoch.D']

        if(pattern1 and pattern2 and pattern3 and pattern4 and pattern5 and pattern6):
            #print(f'Buy the stock {symbol}')
            return symbol.split(':')[1]
        else:
            return None

    def get_all_analysis(self, interval) -> dict:
        analysis = get_multiple_analysis(
            screener=self.screener, 
            interval=self.intervals[interval], 
            symbols=self.symbols
            )
        return analysis

    def get_st_result_list(self, interval) -> list:
        result_list = []
        analysis = self.get_all_analysis(self.intervals[interval])
        try:
            for k in analysis:
                dict_ind = analysis[k].indicators
                temp_symbol = self.ind_st_buy_status(dict_ind, k)
                if temp_symbol != None:
                    result_list.append(temp_symbol)
        except AttributeError:
            print(f"There is no such attribute for {k}")
        return result_list

    def ind_st_buy_signal(self, stock_ind: dict, symbol_pair: dict, symbol: str) -> None:

        current_price = self.get_one_analysis(symbol.split(':')[1], '1m')
        current_price = current_price.indicators['close']

        pattern1 = stock_ind['EMA5'] >= stock_ind['SMA10']
        pattern2 = stock_ind['MACD.macd'] > stock_ind['MACD.signal']
        pattern3 = stock_ind['Ichimoku.BLine'] < current_price
        pattern4 = stock_ind['ADX+DI'] >= stock_ind['ADX-DI']
        pattern5 = stock_ind['SMA10'] < current_price
        pattern6 = stock_ind['Stoch.K'] > stock_ind['Stoch.D']

        all_pattern = (pattern1 and pattern2 and pattern3 and pattern4 and pattern5 and pattern6)

        if all_pattern and symbol_pair[symbol] == -1:
            print(f'INITLIAZE SIGNAL AT NOW BUY {symbol} at {current_price}')
            symbol_pair[symbol] = 1
        elif not all_pattern and symbol_pair[symbol] == -1:
            symbol_pair[symbol] = 0
        elif all_pattern and symbol_pair[symbol] == 0:
            print(f'BUY SIGNAL FOR {symbol} at {current_price}')
            symbol_pair[symbol] = 1
        elif symbol_pair[symbol] == 1 and not all_pattern:
            print(f'SELL SIGNAL FOR {symbol} at {current_price}')
            symbol_pair[symbol] = 0
        
    def listen_live_signal(self, interval, sleep_time=60*15) -> None:
        repeat = 0
        print(f'INTERVAL_1_HOUR {self.screener} INDEX BUY/SELL Signals')
        while True:
            analysis = self.get_all_analysis(self.intervals[interval])
            print(f'Time: {repeat*30} minute')
            repeat += 1
            for k in analysis:
                dict_ind = analysis[k].indicators
                self.ind_st_buy_signal(dict_ind, self.symbols_dict, k)
            time.sleep(sleep_time)  