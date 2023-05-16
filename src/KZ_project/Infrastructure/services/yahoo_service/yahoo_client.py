import yfinance as yf
import pandas as pd

from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.Infrastructure.services.service_client.abstract_service_client import IServiceClient

class YahooClient(IServiceClient):
    def __init__(self, logger: Logger=None):
        self.df = None
        self.logger = logger
            
    def get_history(self, symbol, interval, start, end = None, *args):
        if args:            # Only yahoo download check this data got or not
            df_download = self.yahoo_download(symbol, period=args[0], interval=interval)         
        elif start:            # Only yahoo download check this data got or not
            df_download = self.yahoo_download(symbol, start=start, end=end, interval=interval)     
        return df_download
    
    def yahoo_download(self, symbol, period: str=None, interval: str=None, start:str=None, end:str=None) -> pd.DataFrame():
        """For Yahoo API, get data with period interval and specific date.
        you can check wheter yahoo api for getting finance data.

        Returns:
            DataFrame
        """
        if period != None:
            download_dfy = yf.download(symbol, period=period, interval=interval)
            self.log(f'Get {symbol} data from yahoo period: {period} and interval: {interval}')
        elif start != None:
            download_dfy = yf.download(symbol, start=start, end=end, interval=interval)
            self.log(f'Get {symbol} data from yahoo start date: {start} and {end} also interval: {interval}')
        else:
            self.log(f'Inappropriate period, interval, start or end, Check please!')
            raise ValueError('inappropriate period')
        
        return download_dfy
    
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
    
if __name__ == '__main__':
    y = YahooClient()
    df_d = y.get_history('BTC-USD', '1h', start='2022-09-17', end=None)
    print(df_d)