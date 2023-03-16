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
    
    def ticker(self, symbol: str):
        """
        {'symbol': 'BTCUSDT', 'price': '25121.99000000'}
        """
        return float(self.client.get_symbol_ticker(symbol=symbol)["price"])    

    
if __name__ == '__main__':
    client = Client(api_key = api_key, api_secret = api_secret_key, tld = "com")
        
    print( client.get_symbol_ticker(symbol = "BTCUSDT"))
