from KZ_project.Infrastructure.file_processor.data_checker import DataChecker
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
from KZ_project.ml_pipeline.data_pipeline.feature_extractor import FeatureExtractor
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
    def __init__(
        self, 
        data_creator: DataCreator, 
        data_checker: DataChecker
    ):
        self.data_creator = data_creator
        self.data_checker = data_checker
        
        
    @property
    def interval(self) -> str:
        return self.data_creator.interval
    
    def create_aggregate_indicator_matrix(self):
        """Return Calculated and reindexed sorted Indicators colum dataframe...

        Returns:
            Dataframe
        """
        agg_ind_matrix = self.data_creator.download_ohlc_from_client()
        agg_ind_matrix = self.data_creator.create_datetime_index(agg_ind_matrix)
        agg_ind_matrix = self.data_creator.column_names_preparation(agg_ind_matrix, self.data_creator.range_list)
        indicator_df_result = FactoryIndicatorBuilder.create_indicators_columns(
            agg_ind_matrix, 
            self.data_creator.range_list, 
            logger=self.data_creator.logger
            )  
        agg_ind_matrix = indicator_df_result
        agg_ind_matrix = self.data_creator.reindex_and_sorted_cols(agg_ind_matrix)
        return agg_ind_matrix
        
    def create_aggregate_featured_matrix(self):
        agg_ind_matrix = self.create_aggregate_indicator_matrix()
        feature_extractor = FeatureExtractor(
            agg_ind_matrix, 
            self.data_creator.range_list, 
            self.data_creator.interval, 
            self.data_creator.logger
            )
        feature_extractor.create_featured_matrix()
        self.agg_featured_matrix = feature_extractor.featured_matrix 
        return self.agg_featured_matrix       
    
     
