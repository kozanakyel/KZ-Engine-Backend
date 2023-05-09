from matplotlib import pyplot as plt
import numpy
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from KZ_project.ml_pipeline.ai_model_creator.forecasters.abstract_forecaster import AbstractForecaster


class GridSearchableCV():
        
    @staticmethod
    def bestparams_gridcv(n_estimators_list: list, eta_list: list, 
                            max_depth_list: list, model, 
                            X_train, y_train,
                            verbose: int=0, is_plot: bool=False,
                             ) -> tuple:

        param_grid = dict(max_depth=max_depth_list, n_estimators=n_estimators_list, eta=eta_list)
        tscv = TimeSeriesSplit(n_splits=5)
        print(f'model type: {model.__class__.__name__}')
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="neg_log_loss", n_jobs=-1, cv=tscv,
                verbose=verbose)
        grid_result = grid_search.fit(X_train, y_train)

        if verbose > 0:
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            
        if is_plot:
            means = grid_result.cv_results_['mean_test_score']
            scores = numpy.array(means).reshape(len(max_depth_list), len(n_estimators_list))
            for i, value in enumerate(max_depth_list):
                plt.plot(n_estimators_list, scores[i], label='depth: ' + str(value))
                plt.legend()
                plt.savefig('./data/plots/estimator_best_param.png')
                print('Best estimator plot saved')
        
        return grid_result.best_params_   