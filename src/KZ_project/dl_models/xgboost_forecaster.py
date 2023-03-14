import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance

from KZ_project.logger.logger import Logger
import matplotlib.pyplot as plt
import operator
import itertools 
import os


class XgboostForecaster():

    def __init__(self, n_estimators: int=100, tree_method: str='gpu_hist', eta: float=0.1, 
                    max_depth: int=1, objective: str='binary', eval_metric: str='logloss', 
                    cv: int=0, is_kfold: bool=False, logger: Logger=None):
        self.n_estimators = n_estimators
        self.tree_method = tree_method
        self.eta = eta
        self.max_depth = max_depth
        self.objective = objective
        self.eval_metric = eval_metric
        self.cv = cv
        self.is_kfold = is_kfold
        self.logger = logger
        if self.objective == 'binary':
            self.model = XGBClassifier(n_estimators=self.n_estimators, tree_method=self.tree_method, 
                    eta=self.eta, max_depth=self.max_depth, early_stopping_rounds = 20)
        elif self.objective == 'regression':
            self.model = XGBRegressor(n_estimators=self.n_estimators, tree_method=self.tree_method, 
                    eta=self.eta, max_depth=self.max_depth, early_stopping_rounds = 20)
        else: 
            self.log(f'You must use objective BINARY or REGRESSION')

    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
        

    def create_train_test_data(self, x, y, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, shuffle=False)
        if self.objective == 'binary':  
            self.evalset = [(self.X_train, self.y_train), (self.X_test, self.y_test)] 
        else: 
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
            

    def get_score(self):
        yhat = self.model.predict(self.X_test)
        
        if self.is_kfold:  
            scr = self.kf_cv_scores.mean()
            self.log(f'Score for Mean {self.cv} Kfold is: {self.kf_cv_scores.mean()}')
        elif self.objective == 'regression' and self.cv > 0:  
            scr = self.cv_scores.mean()
            self.log(f'Score for Mean {self.cv} Cross-Validation is: {self.cv_scores.mean()}')
        elif self.objective == 'regression':
            scr = self.model.score(self.X_train, self.y_train)  
            self.log(f'Score for Train set regression is: {scr}')
        elif self.objective == 'binary':
            scr = accuracy_score(self.y_test, yhat)
            self.log(f'Accuracy Score for binary classification is: {scr}')
        return scr

    def plot_learning_curves(self):
        self.results = self.model.evals_result()
        fig, ax = plt.subplots()
        ax.plot(self.results['validation_0'][self.eval_metric], label='train')
        ax.plot(self.results['validation_1'][self.eval_metric], label='test')
        ax.legend()
        plt.savefig('learning_curves.png')
        self.log('Learning curves ploting and saved')

    def save_model(self, file_name: str):
        self.model.save_model(file_name)

    def load_model(self, file_name: str):
        self.model.load_model(file_name)

    def bestparams_gridcv(self, n_estimators_list: list, eta_list: list, 
                            max_depth_list: list, verbose: int=0, is_plot: bool=False) -> tuple:

        param_grid = dict(max_depth=max_depth_list, n_estimators=n_estimators_list, eta=eta_list)
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(self.model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=tscv,
                verbose=verbose)
        grid_result = grid_search.fit(self.X_train, self.y_train)

        if verbose > 0:
            self.log("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                self.log("%f (%f) with: %r" % (mean, stdev, param))

            
        if is_plot:
            means = grid_result.cv_results_['mean_test_score']
            scores = numpy.array(means).reshape(len(max_depth_list), len(n_estimators_list))
            for i, value in enumerate(max_depth_list):
                plt.plot(n_estimators_list, scores[i], label='depth: ' + str(value))
                plt.legend()
                plt.savefig('./data/plots/estimator_best_param.png')
                self.log('Best estimator plot saved')
        
        return grid_result.best_params_
        

    def get_model_names(self, path_models: str):
        files = os.listdir(path_models)
        list_files = []
        for f in files:
            if f.endswith('.json'):
                list_files.append(f)
        self.log(f'Existing model names are: {list_files}')

    def get_n_importance_features(self, n: int):
        col_list = self.X_train.columns.to_list()
        dict_importance = {col_list[i]: self.model.feature_importances_[i] for i in range(len(col_list))}
        sorted_d = dict(sorted(dict_importance.items(), key=operator.itemgetter(1), reverse=True))
        n_features = dict(itertools.islice(sorted_d.items(), n)) 
        return n_features

    def plot_fature_importance(self):
        fig, ax = plt.subplots(1,1,figsize=(20,30))
        plot_importance(self.model, ax=ax)
        plt.savefig('./data/plots/importance_feature_params.png')