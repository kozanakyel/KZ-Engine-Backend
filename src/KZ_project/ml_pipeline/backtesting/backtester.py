from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import os
import matplotlib.pyplot as plt
from typing import Tuple, List

from KZ_project.ml_pipeline.ai_model_creator.engines.model_engine import ModelEngine
from KZ_project.ml_pipeline.ai_model_creator.forecasters.gridsearchable_cv import GridSearchableCV
from KZ_project.ml_pipeline.ai_model_creator.forecasters.xgboost_binary_forecaster import (
    XgboostBinaryForecaster,
)
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
from KZ_project.ml_pipeline.data_pipeline.sentiment_feature_matrix_pipeline import (
    SentimentFeaturedMatrixPipeline,
)
from KZ_project.Infrastructure.constant import ROOT_PATH, MODEL_STACK_PATH


class Backtester:
    def __init__(self, period: int, data_creator: DataCreator):
        self.data_creator = data_creator
        self.period = period
        self.featured_matrix = self._create_featured_matrix()
        self.backtest_data = []

    def get_period(self) -> int:
        return self.period

    def set_period(self, period: int) -> None:
        self.period = period

    def _create_featured_matrix(self) -> pd.DataFrame:
        pipeline = SentimentFeaturedMatrixPipeline(
            self.data_creator, None, None, is_twitter=False
        )
        featured_matrix = pipeline.create_sentiment_aggregate_feature_matrix()
        # featured_matrix.to_csv(os.path.join("data", "deneme1.csv"))
        return featured_matrix

    def _get_interval_df(self, start_index: int) -> pd.DataFrame:
        end_index = self._get_end_index(start_index)
        df = self.featured_matrix.iloc[start_index:end_index]
        return df

    def _get_end_index(self, start_index: int) -> int:
        ei = 0
        print(f" get interval type: {self.data_creator.interval}")
        if self.data_creator.interval == "1h":
            ei = start_index + self.period * 24
        elif self.data_creator.interval == "1d":
            ei = start_index + self.period
        else:
            raise ValueError(f"End index is not available end_index: {ei}")
        return ei

    def _predict_next_candle(self, df: pd.DataFrame) -> tuple:
        forecaster = XgboostBinaryForecaster(eta=0.3, tree_method="gpu_hist")
        model_engine = ModelEngine(
            self.data_creator.symbol,
            None,
            self.data_creator.source,
            self.data_creator.interval,
            forecaster,
            is_backtest=True,
        )

        dtt, y_pred, bt_json, acc_score = model_engine.create_model_and_strategy_return(
            df
        )

        return (
            str(dtt + timedelta(hours=int(self.data_creator.interval[0]))),
            int(y_pred),
            bt_json,
            acc_score,
        )

    def _predict_next_candle_from_model(self, df: pd.DataFrame, hashtag: str) -> tuple:
        forecaster = XgboostBinaryForecaster(
            early_stopping_rounds=0, tree_method="gpu_hist"
        )
        model_engine = ModelEngine(
            self.data_creator.symbol,
            hashtag,
            self.data_creator.source,
            self.data_creator.interval,
            forecaster,
            is_backtest=True,
        )
        if self.data_creator.interval[-1] == "h":
            model_engine.xgb.load_model(
                os.path.join(
                    ROOT_PATH,
                    MODEL_STACK_PATH + hashtag,
                    "extract_ad_est_10000_BTCUSDT_binance_model_price_1h_feature_numbers_123.json",
                )
            )
        if self.data_creator.interval[-1] == "d":
            model_engine.xgb.load_model(
                os.path.join(
                    ROOT_PATH,
                    MODEL_STACK_PATH + hashtag,
                    "extract_ad_est_11000_AAPL_yahoo_model_price_1d_feature_numbers_123.json",
                )
            )

        y = df.feature_label
        X = df.drop(columns=["feature_label"], axis=1)

        y_pred = model_engine.xgb.model.predict(X)
        acc_score = accuracy_score(y_pred, y)

        return acc_score, X, y, y_pred

    def grid_backtest(self) -> tuple:
        fm = self.featured_matrix.copy()
        y = fm.feature_label
        x = fm.drop(columns=["feature_label"], axis=1)
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
            verbose=3,
        )
        return best_params  # 0.1, 1, 100

    def backtest(self, backtest_counts: int) -> float:
        fm = self.featured_matrix
        # print(self.featured_matrix.log_return)
        true_accuracy = []
        false_accuracy = []
        succes_predict = 0
        c = 0
        for i in range(backtest_counts):
            # if (fm.loc[fm.index[i]]["log_return"] > 0.005) or (fm.loc[fm.index[i]]["log_return"] < -0.005):
            # c= c+1
            # print('condition agregated ', c)
            df = self._get_interval_df(i)
            # df.to_csv(f"./data/outputs/model_fine_tuned_data/prompt_{i}.csv")
            dt, signal, bt, accuracy_score = self._predict_next_candle(df)
            ei = self._get_end_index(i)
            actual_result = (fm.loc[fm.index[ei + 1]]["log_return"] > 0).astype(int)
            # log_return = fm.loc[fm.index[ei+1]]["log_return"]

            self.backtest_data.append(
                (
                    dt,
                    accuracy_score,
                    signal,
                    actual_result,
                    fm.loc[fm.index[ei + 1]]["log_return"],
                )
            )

            if actual_result == signal:
                succes_predict = succes_predict + 1
                true_accuracy.append(accuracy_score)
            else:
                false_accuracy.append(accuracy_score)
        return succes_predict / len(self.backtest_data)


