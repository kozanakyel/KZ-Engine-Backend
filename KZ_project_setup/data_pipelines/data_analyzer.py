import os
from abc import ABC, abstractmethod



class DataSaver(ABC):
    
    @abstractmethod
    def save_data(self):
        pass
    
class CSVDataSaver(DataSaver):
    """ this class save pure, output ind, and features,
        get main, pure and featured path,
        MAybe you can use default path,
        if not exist the others ways
    """
    def __init__(self, main_path: str, pure_path: str, feature_path: str, prefix_path: str):
        self.prefix_path = prefix_path
        self.main_path = main_path
        self.pure_path = pure_path
        self.feature_path = feature_path
        
      
    def save_data(self) -> None:
        
        return super().save_data()

class DataChecker(ABC):
    def __init__(self, symbol):
        self.symbol = symbol
        
    
    
       

class FolderDataChecker(DataChecker):
    def __init__(self, symbol: str, main_path: str, pure_path: str, feature_path: str, prefix_path: str,
                 period: str, interval: str, start_date: str, end_date: str):
        super().__init__(symbol)
        self.prefix_path = prefix_path
        self.main_path = main_path
        self.pure_path = pure_path
        self.feature_path = feature_path
        
        self.main_path_df = prefix_path + self.main_path
        self.pure_path_df = prefix_path + self.pure_path
        self.feature_path_df = prefix_path + self.feature_path
        
    def create_pure(self) -> str:
        return os.path.join(self.pure_path_df, self.symbol)
    
    def create_main(self) -> str:
        return os.path.join(self.main_path_df, self.symbol)
    
    def create_feature(self) -> str:
        return os.path.join(self.feature_path_df, self.symbol)
    
    @property
    def is_pure_exist(self):
        return os.path.exists(self.create_pure_folder())
    
    @property
    def is_main_exist(self):
        return os.path.exists(self.create_main_folder())
    
    @property
    def is_feature_exist(self):
        return os.path.exists(self.create_feature_folder())
        

if __name__ == '__main__':
    MAIN_PATH = '/data/outputs/data_ind/'
    PURE_PATH = '/data/pure_data/'
    FEATURE_PATH = '/data/outputs/feature_data/'
    PREFIX_PATH = './KZ_project_setup'
    SYMBOL = 'BTC-USD' 
    
    d = FolderDataChecker(symbol=SYMBOL, period=None, interval=None, 
                    start_date=None, end_date=None, prefix_path=PREFIX_PATH, 
                    main_path=MAIN_PATH, pure_path=PURE_PATH,
                    feature_path=FEATURE_PATH)  
    
    print(os.getcwd())
    print(d.create_main())
    print(d.is_main_exist)