from KZ_project.Infrastructure.services.binance_service.binance_client import (
    BinanceClient,
)
from dotenv import load_dotenv
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from KZ_project.ml_pipeline.backtesting.backtester import Backtester
from KZ_project.ml_pipeline.backtesting.backtrader_model_return_calculator import BacktraderModelReturnCalculator

from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator

load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret_key = os.getenv("BINANCE_SECRET_KEY")


client = BinanceClient(api_key, api_secret_key)
data_creator = DataCreator(
    symbol="BTCUSDT",
    source="binance",
    range_list=[i for i in range(5, 21)],
    period=None,
    interval="1d",
    start_date="2020-01-01",
    client=client,
)

# for backtest yahoo
# client_yahoo = YahooClient()
# data_creator = DataCreator(
#     symbol="AAPL",
#     source='yahoo',
#     range_list=[i for i in range(5, 21)],
#     period='max',
#     interval="1d",
#     start_date="2023-05-10",
#     client=client_yahoo
# )

backtrader = Backtester(365, data_creator)
bt_model_calculator = BacktraderModelReturnCalculator()
# acc_score, x, y, y_pred = backtrader._predict_next_candle_from_model(backtrader.featured_matrix)
result_score = backtrader.backtest(365)
print(f"backtest result: {result_score}")
# print(f"backtest data: {backtrader.backtest_data}"

bt_model_calculator.calculate_profit_loss_from_backtest_data(backtrader.backtest_data)
dates = [item[0] for item in backtrader.backtest_data]
datetime_objects = [
    datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S%z") for dt_str in dates
]
# print(datetime_objects, type(dates[0]))
cumulative_pl = bt_model_calculator.create_list_profit_loss_from_bt(backtrader.backtest_data)[1:]

fig, ax = plt.subplots()
ax.plot(datetime_objects, cumulative_pl, linestyle="solid")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Profit/Loss ($)")
ax.set_title(
    f"{data_creator.symbol} Cumulative Profit/Loss between {datetime_objects[0]} and {datetime_objects[-1]} daily"
)

# Set the date formatter to display only the hour and minute in the format
# date_format = DateFormatter('%Y-%m-%d')
# ax.xaxis.set_major_formatter(date_format)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# acc_score = backtrader._predict_next_candle_from_model(backtrader.featured_matrix, 'btc')
# print(f'ACCURACY SCORE FOR LAST BACKTEST: {acc_score} last shape: {backtrader.featured_matrix.shape}')
# print(f'ACCURACY SCORE FOR LAST BACKTEST: {acc_score} {x} {y} {y_pred} last shape: {backtrader.featured_matrix.shape}')
# backtrader.create_retuns_data(x, y_pred)
# bt_json = backtrader.trade_fee_net_returns(x)

# # Assuming self.backtest_data is a list of tuples
# data = pd.DataFrame(bt.backtest_data, columns=['date', 'accuracy', 'signal', 'actual'])
# data['date'] = pd.to_datetime(data['date'])

# # Plot the data
# plt.plot(data['date'], data['accuracy'], label='Accuracy')
# # plt.plot(data['date'], data['signal'], label='Signal')
# # plt.plot(data['date'], data['actual'], label='Actual')
# plt.legend()
# plt.show()

# gcv = backtrader.grid_backtest()
# print(f'best params: {gcv}')  # 0.5, 1, 400
# best params: {'eta': 0.9, 'max_depth': 1, 'n_estimators': 1200}
