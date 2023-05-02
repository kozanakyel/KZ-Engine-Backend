from KZ_project.ml_pipeline.indicators.pandasta_indicator import PandasTaIndicator
from KZ_project.ml_pipeline.indicators.talib_indicator import TalibIndicator


class FactoryIndicatorBuilder():
    
    @staticmethod
    def create_indicators_columns(df, range_list, logger) -> None:
        result_df = None
        try:
            import talib
            print('Start TA-LIB module')
            talib_indicator = TalibIndicator(df, range_list, logger)
            talib_indicator.create_ind_cols()
            result_df = talib_indicator.df.copy()
            print('created indicators columns with TA-LIB')
        except ModuleNotFoundError:
            print('Start Pandas_ta module')
            ta_indicator = PandasTaIndicator(df, range_list, logger)
            ta_indicator.create_ind_cols()
            result_df = ta_indicator.df.copy()
            print('created indicators columns with Pandas_ta') 
        return result_df
            
        