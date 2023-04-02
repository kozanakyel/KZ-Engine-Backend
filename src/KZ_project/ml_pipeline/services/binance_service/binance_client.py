import pandas as pd
from binance.client import Client
from binance import ThreadedWebsocketManager

from KZ_project.logger.logger import Logger

# binance API icin zaman sorunsalini wsl2 da cozer
# sudo apt install ntpdate
# sudo ntpdate -sb time.nist.gov


class BinanceClient():
    
    def __init__(self, api_key: str, api_secret_key: str, logger: Logger=None):
        self.api_key = api_key
        self.api_secret_key = api_secret_key
        self.logger = logger
        
        self.client = Client(api_key = self.api_key, 
                             api_secret = self.api_secret_key, 
                             tld = "com")
        self.twm = ThreadedWebsocketManager() 
        
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
    
    def account_balances(self) -> pd.DataFrame():
        account = self.client.get_account()
        df = pd.DataFrame(account["balances"])
        df.free = pd.to_numeric(df.free, errors="coerce")  
        self.log(f'Get account balances for {self}') 
        return df.loc[df.free > 0]
    
    
    def get_exchange_info(self) -> dict:
        return self.client.get_exchange_info() 
    
    def get_trade_fee(self, symbol: str) -> list:
        """
        example symbol: "BTCEUR"
        return [dict]  only one dict element
        """
        return self.client.get_trade_fee(symbol = symbol)
    
    def ticker_price(self, symbol: str):
        """
        {'symbol': 'BTCUSDT', 'price': '25121.99000000'}
        """
        return float(self.client.get_symbol_ticker(symbol=symbol)["price"])  
    
    def ticker_filter(self, filter_keys: list=[]) -> pd.DataFrame():
        # use for list ["USD", "BTC", etc]
        prices = self.client.get_all_tickers()
        df_tickers = pd.DataFrame(prices)
        
        if len(filter_keys) != 0:
            index_list = []
            for i in filter_keys:
                conditions = df_tickers.symbol.str.contains(i)
                index_list.append(df_tickers[conditions].index)
            final_index = set.intersection(*map(set, index_list))
            return df_tickers.loc[final_index]
        else:
            return df_tickers  
        
    def ticker_daily(self, symbol: str) -> dict:
        # pd.to_datetime(last24["closeTime"], unit = "ms") 
        return self.client.get_ticker(symbol=symbol)
    
    def get_history(self, symbol, interval, start, end = None):
        """
        valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        example: start = "2021-10-01 10:00:00"
        """
        bars = self.client.get_historical_klines(symbol = symbol, interval = interval,
                                        start_str = start, end_str = end, limit = 1000)
        df = pd.DataFrame(bars)
        df["Datetime"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC')

        # Format the datetime column to your desired format
        df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
        
        
        df.columns = ["Open Time", "open", "high", "low", "close", "volume",
                  "Clos Time", "Quote Asset Volume", "Number of Trades",
                  "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Datetime"]
        df = df[["Datetime", "open", "high", "low", "close", "volume"]].copy()
        df["adj_close"] = df["close"]
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace = True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
    
        self.log(f'Fetch data from Binance API with OHLC from {start} and inbterval {interval}')
        return df  
    
 

        
        
    
    


