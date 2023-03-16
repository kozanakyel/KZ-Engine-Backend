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
        
    
if __name__ == '__main__':
    client = Client(api_key = api_key, api_secret = api_secret_key, tld = "com")
    account = client.get_account()

    print(account.keys())
