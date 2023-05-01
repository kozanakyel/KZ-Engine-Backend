import abc
import pandas as pd


class IBacktestable(abc.ABC):
    
    @abc.abstractmethod
    def create_retuns_data(self, X_pd, y_pred):
        raise NotImplementedError
        
    @abc.abstractmethod   
    def trade_fee_net_returns(self, X_pd: pd.DataFrame()):
        raise NotImplementedError