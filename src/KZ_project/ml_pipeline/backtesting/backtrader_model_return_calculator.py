from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from KZ_project.core.interfaces.Ifee_calculateable import IFeeCalculateable
from KZ_project.core.interfaces.Ireturn_data_creatable import IReturnDataCreatable
from typing import List, Tuple


class BacktraderModelReturnCalculator(IFeeCalculateable, IReturnDataCreatable):
    def create_retuns_data(self, X_pd, y_pred):
        X_pd["position"] = [y_pred[i] for i, _ in enumerate(X_pd.index)]
        X_pd["strategy"] = X_pd.position.shift(1) * X_pd["log_return"]
        X_pd[["log_return", "strategy"]].sum().apply(np.exp)
        X_pd["cstrategy"] = X_pd["strategy"].cumsum().apply(np.exp)
        X_pd["creturns"] = X_pd.log_return.cumsum().apply(np.exp)

    def trade_fee_net_returns(self, X_pd: pd.DataFrame()):
        X_pd["trades"] = X_pd.position.diff().fillna(0).abs()
        commissions = 0.00075  # reduced Binance commission 0.075%
        other = 0.0001  # proportional costs for bid-ask spread & slippage (more detailed analysis required!)
        ptc = np.log(1 - commissions) + np.log(1 - other)

        X_pd["strategy_net"] = (
            X_pd.strategy + X_pd.trades * ptc
        )  # strategy returns net of costs
        X_pd["cstrategy_net"] = X_pd.strategy_net.cumsum().apply(np.exp)

        X_pd[["creturns", "cstrategy", "cstrategy_net"]].plot(
            figsize=(12, 8), title=f"{self.data_creator.symbol}"
        )
        plt.show()
        return X_pd[["creturns", "cstrategy", "cstrategy_net"]]
    
    def calculate_profit_loss_from_backtest_data(self, backtest_data: List[Tuple]):
        total_profit_loss = 0

        for item in backtest_data:
            _, _, signal, _, log_return = item
            profit_loss = signal * log_return
            total_profit_loss += profit_loss

        print("Total Profit/Loss:", total_profit_loss)


    def create_list_profit_loss_from_bt(self, backtest_data: List[Tuple]) -> List:
        cumulative_profit_loss = [0]  # Initialize with 0, as starting balance is $100
        for item in backtest_data:
            _, _, signal, _, log_return = item
            profit_loss = signal * log_return
            cumulative_profit_loss.append(cumulative_profit_loss[-1] + profit_loss)
        return cumulative_profit_loss
    
    def create_list_actual_profit_loss_from_bt(self, backtest_data: List[Tuple]) -> List:
        cumulative_profit_loss = [0]  # Initialize with 0, as starting balance is $100
        for item in backtest_data:
            _, _, signal, act, log_return = item
            profit_loss = 1*log_return  # Treat negative log_return as loss
            cumulative_profit_loss.append(cumulative_profit_loss[-1] + profit_loss)
        return cumulative_profit_loss