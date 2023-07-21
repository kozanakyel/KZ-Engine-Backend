import pandas as pd
from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.core.interfaces.Ibinary_feature_label import IBinaryFeatureLabel
from KZ_project.Infrastructure.file_processor.data_checker import DataChecker
from KZ_project.core.interfaces.Iclient_service import IClientService
from KZ_project.ml_pipeline.japanese_candlestick.japanese_candlestick_creator import JapaneseCandlestickCreator


class DataCreator(IBinaryFeatureLabel):
    def __init__(
        self, 
        symbol: str, 
        source: str, 
        range_list: list, 
        period: str=None, 
        interval: str=None, 
        start_date: str=None, 
        end_date: str=None, 
        scale: int=1, 
        saved_to_csv: bool=False,
        logger: Logger=None, 
        data_checker: DataChecker=None, 
        client: IClientService=None
    ):  
        self.symbol = symbol
        self.source = source
        self.scale = scale
        self.range_list = [i*scale for i in range_list]
        self.period = period
        self.interval = interval 
        self.start_date = start_date
        self.end_date = end_date
        self.client = client
        self.data_checker = data_checker
        self.saved_to_csv = saved_to_csv
        self.logger = logger
        self.df = None
    
    def download_ohlc_from_client(self):
        if self.period:            # Only yahoo download check this data got or not
            df_download = self.client.get_history(self.symbol, self.interval, None, None, self.period)             
        else:
            df_download = self.client.get_history(symbol = self.symbol, interval = self.interval,
                                                  start = self.start_date, 
                                                  end = self.end_date)  
        return df_download  
    
    def create_datetime_index(self, dframe):
        dframe['Datetime'] = dframe.index
        dframe = dframe.set_index('Datetime')
        dframe.dropna(inplace=True)
        
        return dframe
    
    def column_names_preparation(self, dframe, range_list):
        if dframe.shape[0] > range_list[-1]:
            dframe.columns = dframe.columns.str.lower()
            if 'adj close' in dframe.columns.to_list():
                dframe = dframe.rename(columns={'adj close': 'adj_close'})  
            # self.pure_df = dframe.copy()
        else:
            self.log(f'{self.symbol} shape size not sufficient from download')
            raise ValueError('shape size not sufficient from download')
        return dframe
    
    
    def create_binary_feature_label(self, df: pd.DataFrame()) -> None:
        """Next candle Binary feature actual reason forecasting to this label

        Args:
            df (pd.DataFrame):
        """
        df['feature_label'] = (df['log_return'] > 0).astype(int)
        df['feature_label'] = df['feature_label'].shift(-1)
        df["feature_label"][df.index[-1]] = df["feature_label"][df.index[-2]]
        
    def reindex_and_sorted_cols(self, dframe):
        dframe.columns = dframe.columns.str.lower()              # Columns letters to LOWER
        dframe = dframe.reindex(sorted(dframe.columns), axis=1) # sort Columns with to alphabetic order
        self.create_binary_feature_label(dframe)                  # Add next Candle binary feature
        dframe.dropna(inplace= True, how='any')
        self.log(f'Created Feature label for next day and dropn NaN values')
        return dframe
    
    def get_candlesticks(self, is_complete=False):
        """_summary_

        Args:
            dframe (Dataframe): 

        Returns:
            dataframe: added candlestick structure and labels for last 20 timestamp
        """
        df = self.download_ohlc_from_client()
        df = self.create_datetime_index(df)
        df = self.column_names_preparation(df, self.range_list)
        if is_complete: df = df.copy()
        else: df = df.tail(20).copy()    # directly default value
        JapaneseCandlestickCreator.create_candle_columns(df)
        JapaneseCandlestickCreator.create_candle_label(df) 
        return df
    
        
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
            
if __name__ == '__main__':
    from KZ_project.Infrastructure.services.yahoo_service.yahoo_client import YahooClient
    MAIN_PATH = '/data/outputs/data_ind/'
    PURE_PATH = '/data/pure_data/'
    FEATURE_PATH = '/data/outputs/feature_data/'
    PREFIX_PATH = '.'
    SYMBOL = 'BTC-USD' 
    PERIOD = "1y"
    INTERVAL = '1d'
    START_DATE = '2021-06-30'
    END_DATE = '2022-07-01'
    HASHTAG = 'btc'
    scale = 1
    range_list = [i for i in range(5,21)]
    range_list = [i*scale for i in range_list]
    source='yahoo'

    y = YahooClient()
    d = DataCreator(symbol=SYMBOL, source=source, range_list=range_list, period=PERIOD, interval=INTERVAL, 
                    start_date=START_DATE, end_date=END_DATE, 
                    client=y)  
    df = d.download_ohlc_from_client()
    df = d.create_datetime_index(df)
    df = d.column_names_preparation(df, range_list)
    candle_japon = d.get_current_candlestick(df)
    print(candle_japon)