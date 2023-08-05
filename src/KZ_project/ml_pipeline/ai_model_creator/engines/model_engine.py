from datetime import timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json
from KZ_project.Infrastructure.services.yahoo_service.yahoo_client import YahooClient
from KZ_project.core.domain.abstract_forecaster import AbstractForecaster
from KZ_project.core.interfaces.Ifee_calculateable import IFeeCalculateable
from KZ_project.core.interfaces.Ireturn_data_creatable import IReturnDataCreatable
from KZ_project.ml_pipeline.ai_model_creator.forecasters.xgboost_binary_forecaster import (
    XgboostBinaryForecaster,
)

from KZ_project.webapi.services import services
from KZ_project.webapi.entrypoints.flask_app import get_session


class ModelEngine(IFeeCalculateable, IReturnDataCreatable):
    def __init__(
        self,
        symbol: str,
        symbol_cut: str,
        source: str,
        interval: str,
        forecaster: AbstractForecaster,
        is_backtest: bool = False,
    ):
        self.symbol = symbol
        self.source = source
        self.interval = interval
        self.symbol_cut = symbol_cut
        self.is_backtest = is_backtest
        self.xgb = forecaster

    def create_retuns_data(self, X_pd, y_pred):
        X_pd["position"] = [y_pred[i] for i, _ in enumerate(X_pd.index)]
        X_pd["strategy"] = X_pd.position.shift(1) * X_pd["log_return"]
        X_pd[["log_return", "strategy"]].sum().apply(np.exp)
        X_pd["cstrategy"] = X_pd["strategy"].cumsum().apply(np.exp)
        X_pd["creturns"] = X_pd.log_return.cumsum().apply(np.exp)

        valuable_features = self.xgb.get_n_importance_features()
        index_col = X_pd.index

        X_pd["importance_features"] = ""
        len_index = len(index_col)
        len_features = len(valuable_features)

        if len_index >= len_features:
            len_cols = len_features
        else:
            len_cols = len_index

        for i in range(len_cols):
            X_pd.at[index_col[i], "importance_features"] = valuable_features[i]

    def trade_fee_net_returns(self, X_pd: pd.DataFrame()):
        X_pd["trades"] = X_pd.position.diff().fillna(0).abs()
        commissions = 0.00075  # reduced Binance commission 0.075%
        other = 0.0001  # proportional costs for bid-ask spread & slippage (more detailed analysis required!)
        ptc = np.log(1 - commissions) + np.log(1 - other)

        X_pd["strategy_net"] = (
            X_pd.strategy + X_pd.trades * ptc
        )  # strategy returns net of costs
        X_pd["cstrategy_net"] = X_pd.strategy_net.cumsum().apply(np.exp)

        # X_pd[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8),  title = f"{self.symbol} - Forecaster strategy")
        # plt.show()
        return X_pd[
            ["creturns", "cstrategy", "cstrategy_net", "importance_features"]
        ].to_json()

    def create_model_and_strategy_return(self, df_final: pd.DataFrame()):
        y = df_final.feature_label
        X = df_final.drop(columns=["feature_label"], axis=1)

        self.ai_type = self.xgb.__class__.__name__
        self.xgb.create_train_test_data(X, y, test_size=0.2)

        self.xgb.fit()

        self.model_name = f"extract_ad_est_{self.xgb.model.n_estimators}_{self.symbol}_{self.source}_model_price_{self.interval}_feature_numbers_{X.shape[1]}.json"

        score = self.xgb.get_score()

        print(f"Accuracy Score: {score} and shape: {X.shape}")
        # best_params = GridSearchableCV.bestparams_gridcv([100, 200], [0.1], [1, 3], verbose=3)

        xtest = self.xgb.X_test  # last addeded tro backtest data for modeliing hourly
        ytest = self.xgb.y_test

        if not self.is_backtest:
            if self.interval[-1] == "h":
                datetime_t = str(
                    xtest.index[-1] + timedelta(hours=int(self.interval[0]))
                )
            elif self.interval[-1] == "d":
                datetime_t = str(
                    xtest.index[-1] + timedelta(days=int(self.interval[0]))
                )

            res_str = services.save_crypto_forecast_model_service(
                score,
                get_session(),
                self.symbol_cut,
                self.symbol,
                self.source,
                X.shape[1],
                self.model_name,
                self.interval,
                self.ai_type,
                datetime_t,
            )
            print(f"model engine model save: {res_str}")

        # self.xgb.save_model(f"./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/{self.symbol_cut}/{self.model_name}")
        y_pred = self.xgb.model.predict(xtest)
        # self.create_retuns_data(xtest, y_pred)
        # bt_json = self.trade_fee_net_returns(xtest)
        bt_json = self.xgb.get_n_importance_features()[:10]
        # print(type(bt_json), bt_json)

        print(f"Accuracy Score: {score} last datetime_t: {X.index[-1]}")

        return xtest.index[-1], y_pred[-1], bt_json, score

    def create_model_and_prediction(self, df_final: pd.DataFrame()) -> tuple:
        y = df_final.feature_label
        X = df_final.drop(columns=["feature_label"], axis=1)

        self.ai_type = self.xgb.__class__.__name__
        self.xgb.create_train_test_data(X, y, test_size=0.2)

        self.xgb.fit()

        self.model_name = f"test_{self.symbol}_{self.source}_model_price_{self.interval}_feature_numbers_{X.shape[1]}.json"

        accuracy_score = self.xgb.get_score()

        print(f"Accuracy Score: {accuracy_score} last datetime_t: {X.index[-1]}")
        xtest = self.xgb.X_test  # last addeded tro backtest data for modeliing hourly
        ytest = self.xgb.y_test
        y_pred = self.xgb.model.pr
        return xtest, ytest, accuracy_score, xtest.index[-1], ytest[-1]

    def get_strategy_return(self, xtest, ytest):
        self.create_retuns_data(xtest, ytest)
        bt_json = self.trade_fee_net_returns(xtest)
        return json.dumps(bt_json)

    def save_model_to_service(self, xtest, acc_score, x_shape):
        if self.interval[-1] == "h":
            datetime_t = str(xtest.index[-1] + timedelta(hours=int(self.interval[0])))
        elif self.interval[-1] == "d":
            datetime_t = str(xtest.index[-1] + timedelta(days=int(self.interval[0])))

        res_str = services.save_crypto_forecast_model_service(
            acc_score,
            get_session(),
            self.symbol_cut,
            self.symbol,
            self.source,
            x_shape.shape[1],
            self.model_name,
            self.interval,
            self.ai_type,
            datetime_t,
        )
        return res_str.status


if __name__ == "__main__":
    from KZ_project.Infrastructure.services.binance_service.binance_client import (
        BinanceClient,
    )
    from dotenv import load_dotenv
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
    from KZ_project.ml_pipeline.data_pipeline.sentiment_feature_matrix_pipeline import (
        SentimentFeaturedMatrixPipeline,
    )

    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret_key = os.getenv("BINANCE_SECRET_KEY")

    client = BinanceClient(api_key, api_secret_key)
    data_creator = DataCreator(
        symbol="BTCUSDT",
        source="binance",
        range_list=[i for i in range(5, 21)],
        period=None,
        interval="1h",
        start_date="2018-01-15",
        end_date="2023-01-01",
        client=client,
    )

    # client_yahoo = YahooClient()
    # data_creator = DataCreator(
    #     symbol="AAPL",
    #     source='yahoo',
    #     range_list=[i for i in range(5, 21)],
    #     period='max',
    #     interval="1d",
    #     start_date="2018-01-01",
    #     end_date="2023-01-01",
    #     client=client_yahoo
    # )

    forecaster = XgboostBinaryForecaster(
        n_estimators=7000,
        eta=0.3,
        max_depth=1,
        early_stopping_rounds=0,
        cv=5,
        is_kfold=True,
    )
    model_engine = ModelEngine(
        data_creator.symbol,
        "btc",
        data_creator.source,
        data_creator.interval,
        forecaster,
        is_backtest=True,
    )
    pipeline = SentimentFeaturedMatrixPipeline(
        data_creator, None, None, is_twitter=False
    )
    featured_matrix = pipeline.create_sentiment_aggregate_feature_matrix()
    dtt, y_pred, bt_json, acc_score = model_engine.create_model_and_strategy_return(
        featured_matrix
    )
