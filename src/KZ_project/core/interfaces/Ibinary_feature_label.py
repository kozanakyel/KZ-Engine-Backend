import abc

import pandas as pd


class IBinaryFeatureLabel(abc.ABC):
    
    @abc.abstractmethod
    def create_binary_feature_label(self, df: pd.DataFrame()) -> None:
        """Next candle Binary feature actual reason forecasting to this label

        Args:
            df (pd.DataFrame):
        """  
        raise NotImplementedError   