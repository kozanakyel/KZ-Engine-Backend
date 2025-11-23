import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt

from KZ_project.Infrastructure.services.binance_service.binance_client import BinanceClient
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
from KZ_project.Infrastructure.constant import DATA_PATH


def the_most(dataframe, percent=2, length=8, MAtype=1) -> pd.DataFrame:
    import talib.abstract as ta
    df = dataframe.copy()
    mostvalue = 'MOST_' + str(length) + '_' + str(percent)
    mavalue = 'MA_' + str(length) + '_' + str(percent)
    trend = 'Trend_' + str(length) + '_' + str(percent)
    # Compute basic upper and lower bands
    if MAtype == 1:
        df[mavalue] = ta.EMA(df, timeperiod=length)
    elif MAtype == 2:
        df[mavalue] = ta.DEMA(df, timeperiod=length)
    elif MAtype == 3:
        df[mavalue] = ta.T3(df, timeperiod=length)
    df['basic_ub'] = df[mavalue] * (1 + percent / 100)
    df['basic_lb'] = df[mavalue] * (1 - percent / 100)
    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(length, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                         df[mavalue].iat[i - 1] > df['final_ub'].iat[i - 1] else \
            df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                         df[mavalue].iat[i - 1] < df['final_lb'].iat[i - 1] else \
            df['final_lb'].iat[i - 1]
    # Set the MOST value
    df[mostvalue] = 0.00
    for i in range(length, len(df)):
        df[mostvalue].iat[i] = df['final_ub'].iat[i] if df[mostvalue].iat[i - 1] == df['final_ub'].iat[i - 1] and \
                                                        df[mavalue].iat[i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[mostvalue].iat[i - 1] == df['final_ub'].iat[i - 1] and df[mavalue].iat[i] > \
                                     df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[mostvalue].iat[i - 1] == df['final_lb'].iat[i - 1] and df[mavalue].iat[i] >= \
                                         df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[mostvalue].iat[i - 1] == df['final_lb'].iat[i - 1] and df[mavalue].iat[
                        i] < df['final_lb'].iat[i] else 0.00
    # Mark the trend direction up/down
    df[trend] = np.where((df[mostvalue] > 0.00), np.where((df[mavalue] < df[mostvalue]), 'down', 'up'), np.NaN)
    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    df.fillna(0, inplace=True)
    return df


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret_key = os.getenv('BINANCE_SECRET_KEY')

    client = BinanceClient(api_key, api_secret_key)
    dataf = client.get_history(symbol="BTCUSDT", interval="1d", start="2022-12-01")

    data_creator = DataCreator(
        symbol="BTCUSDT",
        source='binance',
        range_list=[i for i in range(5, 21)],
        period=None,
        interval="1h",
        start_date="2023-02-01",
        client=client
    )

    df = the_most(dataf, MAtype=1)
    df = df.iloc[30:].copy()

    up_to_down = (df['Trend_8_2'].shift(1) == 'up') & (df['Trend_8_2'] == 'down')
    down_to_up = (df['Trend_8_2'].shift(1) == 'down') & (df['Trend_8_2'] == 'up')

    df.plot(y=["close", "MA_8_2", "MOST_8_2"], title='BTCUSDT MOST indicator')

    plt.scatter(df.index[up_to_down], df['close'][up_to_down], marker='^', color='green', label='Buy')
    plt.scatter(df.index[down_to_up], df['close'][down_to_up], marker='v', color='red', label='Sell')

    plt.legend()
    plt.savefig(os.path.join(DATA_PATH, 'most_plot.png'))
