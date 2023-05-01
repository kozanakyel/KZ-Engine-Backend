import abc


class AbstractForecaster(abc.ABC):
    
    @abc.abstractmethod
    def create_train_test_data(self, x, y, test_size):
        raise NotImplementedError
    
    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def save_model(self, file_name: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_model(self, file_name: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_score(self):
        raise NotImplementedError