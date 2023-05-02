from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import json

from KZ_project.ml_pipeline.ai_model_creator.engines.Ibacktestable import IBacktestable
from KZ_project.ml_pipeline.ai_model_creator.forecasters.xgboost_binary_forecaster import XgboostBinaryForecaster
from KZ_project.webapi.services import services
from KZ_project.webapi.entrypoints.flask_app import get_session

class ModelEngine(IBacktestable):
    
    def __init__(self, symbol, symbol_cut, source, interval):
        self.symbol = symbol
        self.source = source
        self.interval = interval
        self.symbol_cut = symbol_cut
        self.data_plot_path = f'./data/plots/model_evaluation/'
        self.model_plot_path = self.data_plot_path + f'{self.symbol_cut}/{self.symbol}_{self.source}_{self.interval}_model_backtest.png'
        self.model_importance_feature = self.data_plot_path + f'{self.symbol_cut}/{self.symbol}_{self.source}_{self.interval}_model_importance.png'
    
    
    
    def create_retuns_data(self, X_pd, y_pred):
        X_pd["position"] = [y_pred[i] for i, _ in enumerate(X_pd.index)]    
        X_pd["strategy"] = X_pd.position.shift(1) * X_pd["log_return"]
        X_pd[["log_return", "strategy"]].sum().apply(np.exp)
        X_pd["cstrategy"] = X_pd["strategy"].cumsum().apply(np.exp) 
        X_pd["creturns"] = X_pd.log_return.cumsum().apply(np.exp) 
        
    def trade_fee_net_returns(self, X_pd: pd.DataFrame()):    
        X_pd["trades"] = X_pd.position.diff().fillna(0).abs()    
        commissions = 0.00075 # reduced Binance commission 0.075%
        other = 0.0001 # proportional costs for bid-ask spread & slippage (more detailed analysis required!)
        ptc = np.log(1 - commissions) + np.log(1 - other)
    
        X_pd["strategy_net"] = X_pd.strategy + X_pd.trades * ptc # strategy returns net of costs
        X_pd["cstrategy_net"] = X_pd.strategy_net.cumsum().apply(np.exp)
    
        X_pd[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8),  title = f"{self.symbol} - Buy and Hold")
        plt.savefig(self.model_plot_path)
        return X_pd[["creturns", "cstrategy", "cstrategy_net"]].to_json()
        
    def get_accuracy_score_for_xgboost_fit_separate_dataset(self, df_final: pd.DataFrame()):
        y = df_final.feature_label
        X = df_final.drop(columns=['feature_label'], axis=1)

        xgb = XgboostBinaryForecaster(n_estimators=500, eta=0.01, max_depth=7, 
                    tree_method='gpu_hist', eval_metric='logloss')
        self.ai_type = xgb.__class__.__name__
        xgb.create_train_test_data(X, y, test_size=0.2)
        
        xgb.fit()
    
        self.model_name = f'test_{self.symbol}_{self.source}_model_price_{self.interval}_feature_numbers_{X.shape[1]}.json'
        
        # try for instant model evaluation for one week    
        xgb.save_model(f'./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/{self.symbol_cut}/{self.model_name}')
        score = xgb.get_score()

        print(f'First score: {score}')
        #xgb.plot_learning_curves()
        # best_params = GridSearchableCV.bestparams_gridcv([100, 200], [0.1], [1, 3], verbose=3)

        # modelengine works
        ytest = xgb.y_test
        ypred_reg = xgb.model.predict(xgb.X_test)
        print(f'Last accuracy: {accuracy_score(ytest, ypred_reg)}')
        acc_score = accuracy_score(ytest, ypred_reg)
        print(f'Confusion Matrix:\n{confusion_matrix(ytest, ypred_reg)}')    
        
        res_str = services.save_crypto_forecast_model_service(acc_score, get_session(), self.symbol_cut, 
                                                          self.symbol, self.source, X.shape[1], self.model_name,
                                                          self.interval, self.ai_type)
        print(f'model engine model save: {res_str}')
    
        xgb.plot_feature_importance(
            self.model_importance_feature, 
            self.symbol
            )
    
        xtest = xgb.X_test
        self.create_retuns_data(xtest, ytest)
        bt_json = self.trade_fee_net_returns(xtest)
        print(f'x last row {xtest.index[-1]}\n prediction last candle {ytest[-1]}')
        print(f'bt: method inside: {json.dumps(bt_json)}')
        
        return xtest.index[-1], ytest[-1], json.dumps(bt_json)
        
