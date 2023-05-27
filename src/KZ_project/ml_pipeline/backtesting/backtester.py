from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from KZ_project.Infrastructure.services.yahoo_service.yahoo_client import YahooClient
from KZ_project.core.interfaces.Ifee_calculateable import IFeeCalculateable
from KZ_project.core.interfaces.Ireturn_data_creatable import IReturnDataCreatable

from KZ_project.ml_pipeline.ai_model_creator.engines.model_engine import ModelEngine
from KZ_project.ml_pipeline.ai_model_creator.forecasters.xgboost_binary_forecaster import XgboostBinaryForecaster
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
from KZ_project.ml_pipeline.data_pipeline.sentiment_feature_matrix_pipeline import SentimentFeaturedMatrixPipeline
from KZ_project.Infrastructure.constant import ROOT_PATH, MODEL_STACK_PATH


class Backtester(IFeeCalculateable, IReturnDataCreatable):

    def __init__(
        self,
        period: int,
        data_creator: DataCreator
    ):
        self.data_creator = data_creator
        self.period = period
        self.featured_matrix = self._create_featured_matrix()
        self.backtest_data = []

    def get_period(self) -> int:
        return self.period

    def set_period(self, period: int) -> None:
        self.period = period

    def _create_featured_matrix(self) -> pd.DataFrame:
        pipeline = SentimentFeaturedMatrixPipeline(self.data_creator, None, None, is_twitter=False)
        featured_matrix = pipeline.create_sentiment_aggregate_feature_matrix()
        # featured_matrix.to_csv(os.path.join("data", "deneme1.csv"))
        return featured_matrix

    def _get_interval_df(self, start_index: int) -> pd.DataFrame:
        end_index = self._get_end_index(start_index)
        df = self.featured_matrix.iloc[start_index:end_index]
        return df

    def _get_end_index(self, start_index: int) -> int:
        ei = 0
        print(f' get interval type: {self.data_creator.interval}')
        if self.data_creator.interval == '1h':
            ei = start_index + self.period * 24
        elif self.data_creator.interval == '1d':
            ei = start_index + self.period
        else:
            raise ValueError(f'Enb index is not available end_index: {ei}')
        return ei

    def _predict_next_candle(self, df: pd.DataFrame) -> tuple:
        model_engine = ModelEngine(
            self.data_creator.symbol,
            None,
            self.data_creator.source,
            self.data_creator.interval,
            is_backtest=True
        )

        dtt, y_pred, bt_json, acc_score = model_engine.create_model_and_strategy_return(df)

        return str(dtt + timedelta(hours=int(self.data_creator.interval[0]))), int(y_pred), bt_json, acc_score

    def _predict_next_candle_from_model(
            self,
            df: pd.DataFrame,
            hashtag: str
    ) -> tuple:
        forecaster = XgboostBinaryForecaster(early_stopping_rounds=0)
        model_engine = ModelEngine(
            self.data_creator.symbol,
            hashtag,
            self.data_creator.source,
            self.data_creator.interval,
            forecaster,
            is_backtest=True
        )
        if self.data_creator.interval[-1] == 'h':
            model_engine.xgb.load_model(
                os.path.join(
                    ROOT_PATH,
                    MODEL_STACK_PATH + hashtag,
                    "extract_ad_est_10000_BTCUSDT_binance_model_price_1h_feature_numbers_123.json"
                )
            )
        if self.data_creator.interval[-1] == 'd':
            model_engine.xgb.load_model(
                os.path.join(
                    ROOT_PATH,
                    MODEL_STACK_PATH + hashtag,
                    "extract_ad_est_11000_AAPL_yahoo_model_price_1d_feature_numbers_123.json"
                )
            )

        y = df.feature_label
        X = df.drop(columns=['feature_label'], axis=1)

        y_pred = model_engine.xgb.model.predict(X)
        acc_score = accuracy_score(y_pred, y)

        return acc_score, X, y, y_pred

    def grid_backtest(self) -> tuple:
        fm = self.featured_matrix.copy()
        y = fm.feature_label
        x = fm.drop(columns=['feature_label'], axis=1)
        xgb = XgboostBinaryForecaster(early_stopping_rounds=0)
        xgb.create_train_test_data(x, y, test_size=0.2)
        model_gcv = xgb.model
        best_params = GridSearchableCV.bestparams_gridcv(
            n_estimators_list=[300, 900, 1500, 4000, 8000],
            eta_list=[0.01, 0.03, 0.05, 0.07, 0.1],
            max_depth_list=[1, 3],
            model=model_gcv,
            X_train=xgb.X_train,
            y_train=xgb.y_train,
            verbose=3
        )
        return best_params  # 0.1, 1, 100

    def backtest(self, backtest_counts: int) -> float:
        fm = self.featured_matrix
        true_accuracy = []
        false_accuracy = []
        succes_predict = 0
        for i in range(backtest_counts):
            df = self._get_interval_df(i)
            # df.to_csv(f"./data/outputs/model_fine_tuned_data/prompt_{i}.csv")
            dt, signal, bt, accuracy_score = self._predict_next_candle(df)
            ei = self._get_end_index(i)
            actual_result = (fm.loc[fm.index[ei + 1]]["log_return"] > 0).astype(int)
            # log_return = fm.loc[fm.index[ei+1]]["log_return"]
            # if accuracy_score > 0.50:
            self.backtest_data.append((dt, accuracy_score, signal, actual_result))

            if actual_result == signal:
                succes_predict = succes_predict + 1
                true_accuracy.append(accuracy_score)
            else:
                false_accuracy.append(accuracy_score)
        return succes_predict / len(self.backtest_data)

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

        X_pd["strategy_net"] = X_pd.strategy + X_pd.trades * ptc  # strategy returns net of costs
        X_pd["cstrategy_net"] = X_pd.strategy_net.cumsum().apply(np.exp)

        X_pd[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize=(12, 8), title=f"{self.data_creator.symbol}")
        plt.show()
        return X_pd[["creturns", "cstrategy", "cstrategy_net"]]


if __name__ == '__main__':
    from KZ_project.Infrastructure.services.binance_service.binance_client import BinanceClient
    from KZ_project.ml_pipeline.ai_model_creator.forecasters.gridsearchable_cv import GridSearchableCV
    from dotenv import load_dotenv
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret_key = os.getenv('BINANCE_SECRET_KEY')

    client = BinanceClient(api_key, api_secret_key)
    data_creator = DataCreator(
        symbol="BTCUSDT",
        source='binance',
        range_list=[i for i in range(5, 21)],
        period=None,
        interval="1h",
        start_date="2022-06-01",
        client=client
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

    backtrader = Backtester(7, data_creator)
    # acc_score, x, y, y_pred = backtrader._predict_next_candle_from_model(backtrader.featured_matrix)
    # result_score = bt.backtest(1)
    acc_score = backtrader._predict_next_candle_from_model(backtrader.featured_matrix, 'btc')
    print(f'ACCURACY SCORE FOR LAST BACKTEST: {acc_score} last shape: {backtrader.featured_matrix.shape}')
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
