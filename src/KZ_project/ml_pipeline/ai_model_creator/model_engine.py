from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from KZ_project.core.adapters.crypto_repository import CryptoRepository
from KZ_project.core.adapters.forecastmodel_repository import ForecastModelRepository

from KZ_project.ml_pipeline.ai_model_creator.xgboost_forecaster import XgboostForecaster
from KZ_project.ml_pipeline.data_generator.data_manipulation import DataManipulation
from KZ_project.ml_pipeline.services.twitter_service.tweet_sentiment_analyzer import TweetSentimentAnalyzer

from KZ_project.webapi.services import services
from KZ_project.core.adapters.aimodel_repository import AIModelRepository
from KZ_project.Infrastructure.orm_mapper import orm

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from KZ_project.Infrastructure import config
orm.start_mappers()
get_session = sessionmaker(bind=create_engine(config.get_postgres_uri(), pool_size=50, pool_timeout=60, max_overflow=0))


class ModelEngine():
    
    def __init__(self, symbol, symbol_cut, source, interval) -> None:
        self.symbol = symbol
        self.source = source
        self.interval = interval
        self.symbol_cut = symbol_cut
        
    
    def get_tweet_sentiment_hourly_filed(self, tweet_file):
        #sent_tweets = pd.read_csv('./data/archieve_data/btc_archieve/btc_hourly_sent_score.csv')
        sent_tweets = pd.read_csv(tweet_file)
        #sent_tweets = pd.read_csv('../data/tweets_data/btc/btc_hour.csv')  
        sent_tweets.Datetime = pd.to_datetime(sent_tweets.Datetime)
        
        if self.interval[-1] == 'd':
            print('yes daily model')
            sent_tweets['Datetime'] = sent_tweets['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
            sent_tweets.Datetime = pd.to_datetime(sent_tweets.Datetime)
        sent_tweets.set_index('Datetime', inplace=True, drop=True)
        
        #sent_tweets.Datetime = pd.to_datetime(sent_tweets.index)
        #sent_tweets = sent_tweets.reset_index()
        #sent_tweets.set_index('Datetime', inplace=True, drop=True)
        
        #sent_tweets.index = sent_tweets.index.tz_convert(None) # only hourly
        print(sent_tweets)
        return sent_tweets
    
    def composite_tweet_sentiment_and_data_manipulation(self, data: DataManipulation,
                                                             tsa: TweetSentimentAnalyzer,
                                                         sent_tweets: pd.DataFrame()):

        df_price_ext = data.extract_features()
        print(f'df: {type(df_price_ext.index)} tweet: {type(sent_tweets.index)}')
        
        df_final = tsa.concat_ohlc_compound_score(df_price_ext, sent_tweets)
        del df_price_ext
        #df_final = df_final.loc['2021-01-01':,:].copy()
        df_final = df_final.rename(columns={"compound_total":"twitter_sent_score"})
        df_final.dropna(inplace=True)
        return df_final
    
    def backtest_prediction(self, X_pd, y_pred):
        X_pd["position"] = [y_pred[i] for i, _ in enumerate(X_pd.index)]
    
        print(X_pd[["log_return", "position"]]) 
    
        X_pd["strategy"] = X_pd.position.shift(1) * X_pd["log_return"]
        X_pd[["log_return", "strategy"]].sum().apply(np.exp)
        X_pd["cstrategy"] = X_pd["strategy"].cumsum().apply(np.exp) 
        X_pd["creturns"] = X_pd.log_return.cumsum().apply(np.exp) 

        X_pd.creturns.plot(figsize = (12, 8), title = f"{self.symbol} - Buy and Hold", fontsize = 12)
        plt.show()

        X_pd[["creturns", "cstrategy"]].plot(figsize = (12 , 8), fontsize = 12)
        plt.show()
        
    def trade_fee_net_returns(self, X_pd: pd.DataFrame()):
        val_counts = X_pd.position.value_counts()
        print(f'val counts: {val_counts}')
    
        X_pd["trades"] = X_pd.position.diff().fillna(0).abs()
        trade_val_count = X_pd.trades.value_counts()
        print(f'trade val counts: {trade_val_count}')
    
        commissions = 0.00075 # reduced Binance commission 0.075%
        other = 0.0001 # proportional costs for bid-ask spread & slippage (more detailed analysis required!)
        ptc = np.log(1 - commissions) + np.log(1 - other)
    
        X_pd["strategy_net"] = X_pd.strategy + X_pd.trades * ptc # strategy returns net of costs
        X_pd["cstrategy_net"] = X_pd.strategy_net.cumsum().apply(np.exp)
    
        X_pd[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8))
        plt.show()
        
    def get_accuracy_score_for_xgboost_fit_separate_dataset(self, df_final: pd.DataFrame()):
        y = df_final.feature_label
        X = df_final.drop(columns=['feature_label'], axis=1)
        
        print(f'shapes {X.shape, y.shape}')
    
        #X["twitter_sent_score"] = X["twitter_sent_score"].shift(1)
        #X["twitter_sent_score"][X.index[0]] = 0

        xgb = XgboostForecaster(objective='binary', n_estimators=500, eta=0.01, max_depth=7, 
                    tree_method='gpu_hist', eval_metric='logloss')
        xgb.create_train_test_data(X, y, test_size=0.2)
        
        xgb.fit()
    
        self.model_name = f'test_{self.symbol}_{self.source}_model_price_{self.interval}_feature_numbers_{X.shape[1]}.json'
    
        xgb.save_model(f'./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/{self.symbol_cut}/{self.model_name}')
        score = xgb.get_score()

        print(f'first score: {score}')
        #xgb.plot_learning_curves()
        xgb.get_model_names('./src/KZ_project/ml_pipeline/ai_model_creator/model_stack/')
        #best_params = xgb.bestparams_gridcv([100, 200], [0.1], [1, 3], verbose=3)

        ytest = xgb.y_test
        ypred_reg = xgb.model.predict(xgb.X_test)
        print(f'Last accuracy: {accuracy_score(ytest, ypred_reg)}')
        acc_score = accuracy_score(ytest, ypred_reg)
        print(f'Confusion Matrix: {confusion_matrix(ytest, ypred_reg)}')

        ### Services test
        self.save_service(X, xgb, acc_score)
        ## New Domain servicesss test
        res_str = self.save_crypto_forecast_model_service(X, xgb, acc_score)
        print(f'New test domain servis result is: {res_str}')

        n_feat = xgb.get_n_importance_features(10)
        print(f'importance: {n_feat}')
    
        xgb.plot_fature_importance()
    
        ypr = xgb.model.predict(X)
        self.backtest_prediction(X, ypr)
        self.trade_fee_net_returns(X)
        
    def save_service(self, X: pd.DataFrame(), forecaster: XgboostForecaster, accuracy_score):
        ### Services test
        session = get_session()
        repo = AIModelRepository(session)
        services.add_aimodel(
            self.symbol, self.source,
            X.shape[1], self.model_name, forecaster.__class__.__name__, 
            self.symbol_cut, accuracy_score,
            repo, session
        )
        
    def save_crypto_forecast_model_service(self, X: pd.DataFrame(), 
                                           forecaster: XgboostForecaster,
                                           accuracy_score):
        session = get_session()
        repo = ForecastModelRepository(session)
        repo_cr = CryptoRepository(session)
        try: 
            finding_crypto = services.get_crypto(
                ticker=self.symbol_cut, 
                repo=repo_cr, 
                session=session
                )
    
            services.add_forecast_model(
                self.symbol,
                self.source,
                X.shape[1],
                self.model_name,
                self.interval,
                forecaster.__class__.__name__,
                self.symbol_cut,
                accuracy_score,
                finding_crypto,
                repo,
                session,
            )
        except (services.InvalidName) as e:
            return f'An errror for creating model {self.symbol}'
        return f'Succesfully created model {self.symbol}'
        
        
    def start_model_engine(self, data:DataManipulation,
                            tsa: TweetSentimentAnalyzer, tweet_file):
        sent_tweets = self.get_tweet_sentiment_hourly_filed(tweet_file)
        df_final = self.composite_tweet_sentiment_and_data_manipulation(data, tsa, sent_tweets)
        #df_final = data.extract_features()
        self.get_accuracy_score_for_xgboost_fit_separate_dataset(df_final)
        
if __name__ == '__main__':
    from KZ_project.Infrastructure import config
    asset_config = config.BitcoinConfig()

    tsa = TweetSentimentAnalyzer()
    data = DataManipulation(asset_config.SYMBOL, asset_config.source, 
                        asset_config.range_list, start_date=asset_config.start_date, 
                        end_date=asset_config.end_date, interval=asset_config.interval, scale=asset_config.SCALE, 
                        prefix_path='.', saved_to_csv=False,
                        logger=asset_config.logger, client=asset_config.client)
    df_price = data.df.copy()
    
    model_engine = ModelEngine(asset_config.SYMBOL, asset_config.SYMBOL_CUT, asset_config.source, asset_config.interval)
    
    model_engine.start_model_engine(data, tsa, asset_config.tweet_file_hourly)