class FactoryIndicatorBuilder:
    """
    Factory class for creating indicator columns in a DataFrame.

    This class provides a static method for creating indicator columns in a DataFrame using either the TA-LIB library
    or the Pandas_ta library.

    Attributes:
        None
    """
    
    @staticmethod
    def create_indicators_columns(df, range_list, logger):
        """
        Return the provided DataFrame without calculating TA-based indicators.

        Third-party TA stacks (TA-LIB, pandas_ta, tradingview-ta) have been removed
        to simplify the runtime footprint. This factory now acts as a no-op pass-through
        so downstream feature extraction can continue to operate on the raw OHLC data.

        Args:
            df (pd.DataFrame): The DataFrame for which indicator columns need to be created.
            range_list (list): List of range values for the indicators.
            logger (Logger): Logger object for logging messages.

        Returns:
            pd.DataFrame: DataFrame with the created indicator columns.

        Raises:
            None
        """
        logger.info("Skipping TA indicator generation; returning raw DataFrame")
        return df
            
        