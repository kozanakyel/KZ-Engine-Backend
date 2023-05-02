from KZ_project.ml_pipeline.data_generator.data_checker import DataChecker
from KZ_project.ml_pipeline.data_generator.data_creator import DataCreator
from KZ_project.ml_pipeline.data_generator.feature_extractor import FeatureExtractor
from KZ_project.ml_pipeline.data_generator.file_data_checker import FileDataChecker
from KZ_project.ml_pipeline.indicators.factory_indicator_builder import FactoryIndicatorBuilder

"""
@author: Kozan Ugur AKYEL
@since: 07/08/2022

This file main purpose is that getting finance data,
Creating indicators column for Dataframe, 
Cleaning the data, is necessary or if you want
convert the data to featured extracted data with some specific strategies.
Add lags, add, indicators specific strategis and 
obtain nearly binary matrix, 
some strategies need 3 level layer for labelling.
But our main purposes is that matrix preparation for the
our Forecaster model...
""" 

class FeaturedMatrixPipeline():
    def __init__(self, data_creator: DataCreator, 
                 data_checker: DataChecker):
        self.data_creator = data_creator
        self.data_checker = data_checker
        
    def create_aggregate_featured_matrix(self):
        self.data_creator.df = self.data_creator.download_ohlc_from_client()
        self.data_creator.df = self.data_creator.create_datetime_index(self.data_creator.df)
        self.data_creator.df = self.data_creator.column_names_preparation(self.data_creator.df, self.data_creator.range_list)
        indicator_df_result = FactoryIndicatorBuilder.create_indicators_columns(self.data_creator.df, self.data_creator.range_list, logger=self.data_creator.logger)  
        self.data_creator.df = indicator_df_result
        self.data_creator.df = self.data_creator.reindex_and_sorted_cols(self.data_creator.df)
        self.feature_extractor = FeatureExtractor(self.data_creator.df, self.data_creator.range_list, self.data_creator.interval, self.data_creator.logger)
        self.feature_extractor.create_featured_matrix()
        self.agg_featured_matrix = self.feature_extractor.featured_matrix 




        
if __name__ == '__main__':
    from KZ_project.ml_pipeline.services.yahoo_service.yahoo_client import YahooClient
    MAIN_PATH = '/data/outputs/data_ind/'
    PURE_PATH = '/data/pure_data/'
    FEATURE_PATH = '/data/outputs/feature_data/'
    PREFIX_PATH = '.'
    SYMBOL = 'BTC-USD' 
    PERIOD = "1y"
    INTERVAL = '1h'
    START_DATE = '2021-06-30'
    END_DATE = '2022-07-01'
    scale = 1
    range_list = [i for i in range(5,21)]
    range_list = [i*scale for i in range_list]
    source='yahoo'

    f = FileDataChecker(symbol=SYMBOL, period=PERIOD, interval=INTERVAL, 
                    start_date=START_DATE, end_date=END_DATE, prefix_path=PREFIX_PATH, 
                    main_path=MAIN_PATH, pure_path=PURE_PATH,
                    feature_path=FEATURE_PATH)

    y = YahooClient()
    d = DataCreator(symbol=SYMBOL, source=source, range_list=range_list, period=PERIOD, interval=INTERVAL, 
                    start_date=START_DATE, end_date=END_DATE, 
                    data_checker=f,
                    client=y)  
    
    agg_matrix = FeaturedMatrixPipeline(d, None)
    agg_matrix.create_aggregate_featured_matrix()
    print(agg_matrix.agg_featured_matrix)
