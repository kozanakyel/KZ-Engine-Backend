import itertools
import operator
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor
from KZ_project.Infrastructure.logger.logger import Logger
from KZ_project.ml_pipeline.ai_model_creator.forecasters.abstract_forecaster import AbstractForecaster


class XgboostRegressorForecaster(AbstractForecaster):
    def __init__(self, n_estimators: int=100, tree_method: str='gpu_hist', eta: float=0.1, 
                    max_depth: int=1, eval_metric: str='logloss', 
                    cv: int=0, is_kfold: bool=False, logger: Logger=None):
        self.n_estimators = n_estimators
        self.tree_method = tree_method
        self.eta = eta
        self.max_depth = max_depth
        self._objective = 'regressor'
        self.eval_metric = eval_metric
        self.cv = cv
        self.is_kfold = is_kfold
        self.logger = logger
        self._model = XGBRegressor(n_estimators=self.n_estimators, tree_method=self.tree_method, 
                    eta=self.eta, max_depth=self.max_depth, early_stopping_rounds = 20)
        
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
            
    def create_train_test_data(self, x, y, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, shuffle=False)
        self.evalset = [(self.X_train, self.y_train), (self.X_test, self.y_test)] 
        self.log(f'Creating X_train, X_test, y_train, y_test, evalset') 

    def fit(self):
        self._model.fit(self.X_train, self.y_train, eval_metric=self.eval_metric, eval_set=self.evalset)
        if self.cv > 0 and self.is_kfold:
            kfold = KFold(n_splits=self.cv, shuffle=True)
            self.kf_cv_scores = cross_val_score(self._model, self.X_train, self.y_train, cv=kfold)
            self.log(f'Kfold CV did with {self.cv} fold') 
        elif self.cv > 0:
            self.cv_scores = cross_val_score(self._model, self.X_train, self.y_train, cv=self.cv)
            self.log(f'CV did with {self.cv} cv')
            
    def get_score(self) -> float:
        # yhat = self._model.predict(self.X_test)
        
        if self.is_kfold:  
            scr = self.kf_cv_scores.mean()
            self.log(f'Score for Mean {self.cv} Kfold is: {self.kf_cv_scores.mean()}')
        elif self._objective == 'regression' and self.cv > 0:  
            scr = self.cv_scores.mean()
            self.log(f'Score for Mean {self.cv} Cross-Validation is: {self.cv_scores.mean()}')
        elif self._objective == 'regression':
            scr = self._model.score(self.X_train, self.y_train)  
            self.log(f'Score for Train set regression is: {scr}')
        return scr
    
    def save_model(self, file_name: str):
        self._model.save_model(file_name)

    def load_model(self, file_name: str):
        self._model.load_model(file_name)  
        
    def get_n_importance_features(self, n: int):
        col_list = self.X_train.columns.to_list()
        dict_importance = {col_list[i]: self._model.feature_importances_[i] for i in range(len(col_list))}
        sorted_d = dict(sorted(dict_importance.items(), key=operator.itemgetter(1), reverse=True))
        n_features = dict(itertools.islice(sorted_d.items(), n)) 
        return n_features