import os
from dotenv import load_dotenv

import pandas as pd
from binance.client import Client
import pandas as ta

# binance API icin zaman sorunsalini wsl2 da cozer
# sudo apt install ntpdate
# sudo ntpdate -sb time.nist.gov

load_dotenv()

api_key = os.getenv('BINANCE_API_KEY')
api_secret_key = os.getenv('BINANCE_SECRET_KEY')

class BinanceClient():
    
    def __init__(self, api_key: str, api_secret_key: str):
        self.api_key = api_key
        self.api_secret_key = api_secret_key
        self.client = Client(api_key = self.api_key, 
                             api_secret = self.api_secret_key, 
                             tld = "com")
    
    def account_balances(self) -> pd.DataFrame():
        account = self.client.get_account()
        df = pd.DataFrame(account["balances"])
        df.free = pd.to_numeric(df.free, errors="coerce")   
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
        df.set_index("Datetime", inplace = True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
    
        return df

    
if __name__ == '__main__':
    client = Client(api_key = api_key, api_secret = api_secret_key, tld = "com")
        
    print( client.get_symbol_ticker(symbol = "BTCUSDT"))
