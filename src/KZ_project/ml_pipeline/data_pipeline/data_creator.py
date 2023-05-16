import pandas as pd
from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.ml_pipeline.data_pipeline.Ibinary_feature_label import IBinaryFeatureLabel
from KZ_project.Infrastructure.file_processor.data_checker import DataChecker
from KZ_project.Infrastructure.services.service_client.abstract_service_client import IServiceClient


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
        client: IServiceClient=None
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
        
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)