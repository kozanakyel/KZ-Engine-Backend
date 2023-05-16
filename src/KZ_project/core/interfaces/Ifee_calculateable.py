import abc

import pandas as pd


class IFeeCalculateable(abc.ABC):
    @abc.abstractmethod   
    def trade_fee_net_returns(self, X_pd: pd.DataFrame()):
        raise NotImplementedError