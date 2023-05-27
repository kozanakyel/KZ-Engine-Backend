import numpy as np
import pandas as pd


class TraderService:
    def __init__(self):
        self.trade_values = None
        self.trades = None
        self.position = None
        self.client = None
        self.units = None
        self.symbol = None
        self.prepared_data = None

    def execute_trades(self):
        if self.prepared_data["position"].iloc[-1] == 1:  # if position is long -> go/stay long
            if self.position == 0:
                order = self.client.client.create_order(symbol=self.symbol, side="BUY", type="MARKET",
                                                        quantity=self.units)
                self.report_trade(order, "GOING LONG")  # NEW
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0:  # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = self.client.client.create_order(symbol=self.symbol, side="SELL", type="MARKET",
                                                        quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL")  # NEW
            self.position = 0

    def report_trade(self, order, going):  # NEW

        # extract data from order object
        side = order["side"]
        time = pd.to_datetime(order["transactTime"], unit="ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)

        # calculate trading profits
        self.trades += 1
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units)

        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]), 3)
            cum_profits = round(np.sum(self.trade_values), 3)
        else:
            real_profit = 0
            cum_profits = round(np.sum(self.trade_values[:-1]), 3)

        # print trade report
        print(2 * "\n" + 100 * "-")
        print("{} | {}".format(time, going))
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(time, base_units, quote_units, price))
        print("{} | Profit = {} | CumProfits = {} ".format(time, real_profit, cum_profits))
        print(100 * "-" + "\n")
