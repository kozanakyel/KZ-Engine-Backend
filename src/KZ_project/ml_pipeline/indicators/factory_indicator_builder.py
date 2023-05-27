import pandas as pd

from KZ_project.ml_pipeline.indicators.pandasta_indicator import PandasTaIndicator
from KZ_project.ml_pipeline.indicators.talib_indicator import TalibIndicator


class FactoryIndicatorBuilder:
    """
    Factory class for creating indicator columns in a DataFrame.

    This class provides a static method for creating indicator columns in a DataFrame using either the TA-LIB library
    or the Pandas_ta library.

    Attributes:
        None
    """
    
    @staticmethod
    def create_indicators_columns(df, range_list, logger) -> pd.DataFrame:
        """
        Creates indicator columns in the DataFrame using TA-LIB or Pandas_ta library.

        This method creates indicator columns in the provided DataFrame using either the TA-LIB library
        or the Pandas_ta library based on the availability of the libraries. If TA-LIB is available, it
        uses the TalibIndicator class to create the indicator columns. If TA-LIB is not available, it uses
        the PandasTaIndicator class instead.

        Args:
            df (pd.DataFrame): The DataFrame for which indicator columns need to be created.
            range_list (list): List of range values for the indicators.
            logger (Logger): Logger object for logging messages.

        Returns:
            pd.DataFrame: DataFrame with the created indicator columns.

        Raises:
            None
        """
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
            
        