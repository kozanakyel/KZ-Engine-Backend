import operator
import itertools 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score

from xgboost import XGBClassifier
from xgboost import plot_importance

from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.core.domain.abstract_forecaster import AbstractForecaster


class XgboostBinaryForecaster(AbstractForecaster):

    def __init__(self, n_estimators: int=100, tree_method: str='gpu_hist', eta: float=0.1, 
                    max_depth: int=1, eval_metric: str='logloss', 
                    cv: int=0, is_kfold: bool=False, early_stopping_rounds: int=20, logger: Logger=None):
        self.n_estimators = n_estimators
        self.tree_method = tree_method
        self.eta = eta
        self.max_depth = max_depth
        self._objective = 'binary'
        self.eval_metric = eval_metric
        self.cv = cv
        self.is_kfold = is_kfold
        self.logger = logger
        self.early_stopping_rounds = early_stopping_rounds
        self.model = XGBClassifier(n_estimators=self.n_estimators, tree_method=self.tree_method, 
                    eta=self.eta, max_depth=self.max_depth, early_stopping_rounds = self.early_stopping_rounds)

    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
        

    def create_train_test_data(self, x, y, test_size, shuffle: bool=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle)
        self.evalset = [(self.X_train, self.y_train), (self.X_test, self.y_test)] 
        self.log(f'Creating X_train, X_test, y_train, y_test, evalset') 

    def fit(self):
        self.model.fit(self.X_train, self.y_train, eval_metric=self.eval_metric, eval_set=self.evalset)
        if self.cv > 0 and self.is_kfold:
            kfold = KFold(n_splits=self.cv, shuffle=True)
            self.kf_cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=kfold)
            self.log(f'Kfold CV did with {self.cv} fold') 
        elif self.cv > 0:
            self.cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=self.cv)
            self.log(f'CV did with {self.cv} cv')
            

    def get_score(self) -> float:
        yhat = self.model.predict(self.X_test)
        
        if self.is_kfold:  
            scr = self.kf_cv_scores.mean()
            self.log(f'Score for Mean {self.cv} Kfold is: {self.kf_cv_scores.mean()}')
        elif self._objective == 'binary':
            scr = accuracy_score(self.y_test, yhat)
            self.log(f'Accuracy Score for binary classification is: {scr}')
        return scr

    def plot_learning_curves(self):
        self.results = self.model.evals_result()
        fig, ax = plt.subplots()
        ax.plot(self.results['validation_0'][self.eval_metric], label='train')
        ax.plot(self.results['validation_1'][self.eval_metric], label='test')
        ax.legend()
        # plt.savefig('learning_curves.png')
        plt.show()
        self.log('Learning curves ploting and saved')

    def save_model(self, file_name: str):
        self.model.save_model(file_name)

    def load_model(self, file_name: str):
        self.model.load_model(file_name)        
        

    def get_n_importance_features(self) -> list:
        valuable_features = []
        
        importances = self.model.feature_importances_
        column_names = self.X_train.columns.tolist()

        # Zip feature importances with column names and sort them
        importance_tuples = sorted(zip(column_names, importances), key=lambda x: x[1], reverse=True)

        # Get the first 50 important features
        top_50_features = importance_tuples[:50]

        # Print the first 50 important features with their importances
        for feature, importance in importance_tuples:
            if importance > 0:
                valuable_features.append(feature)
                # print(f"{feature}: {importance}")
        # col_list = self.X_train.columns.to_list()
        # dict_importance = {col_list[i]: self.model.feature_importances_[i] for i in range(len(col_list))}
        # sorted_d = dict(sorted(dict_importance.items(), key=operator.itemgetter(1), reverse=True))
        # n_features = dict(itertools.islice(sorted_d.items(), n)) 
        # return n_features
        return valuable_features

    def plot_feature_importance(self, file_path, symbol):
        fig, ax = plt.subplots(1,1,figsize=(20,30))
        plot_importance(self.model, ax=ax, title=f'{symbol} Feature Importance')
        # plt.savefig(file_path)
        plt.show()