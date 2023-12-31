import os
from KZ_project.Infrastructure.file_processor.data_checker import DataChecker

class FolderDataChecker(DataChecker):
    def __init__(self, symbol: str, main_path: str, pure_path: str, feature_path: str, prefix_path: str):
        super().__init__(symbol=symbol, 
                         prefix_path=prefix_path,
                         main_path=main_path, 
                         pure_path=pure_path, 
                         feature_path=feature_path)
        
        
    def create_pure(self) -> str:
        return os.path.join(self.pure_path_df, self.symbol)
    
    def create_main(self) -> str:
        return os.path.join(self.main_path_df, self.symbol)
    
    def create_feature(self) -> str:
        return os.path.join(self.feature_path_df, self.symbol)
    
    @property
    def is_pure_exist(self):
        return os.path.exists(self.create_pure())
    
    @property
    def is_main_exist(self):
        return os.path.exists(self.create_main())
    
    @property
    def is_feature_exist(self):
        return os.path.exists(self.create_feature())
    
    @property
    def main_path_df(self) -> str:
        return self.prefix_path + self.main_path
    
    @property
    def pure_path_df(self) -> str:
        return self.prefix_path + self.pure_path
    
    @property
    def feature_path_df(self) -> str:
        return self.prefix_path + self.feature_path
        

if __name__ == '__main__':
    MAIN_PATH = '/data/outputs/data_ind/'
    PURE_PATH = '/data/pure_data/'
    FEATURE_PATH = '/data/outputs/feature_data/'
    PREFIX_PATH = './KZ_project'
    SYMBOL = 'BTC-USD' 
    
    d = FolderDataChecker(symbol=SYMBOL, prefix_path=PREFIX_PATH, 
                    main_path=MAIN_PATH, pure_path=PURE_PATH,
                    feature_path=FEATURE_PATH)  
    
    print(os.getcwd())
    print(d.create_main())
    print(d.is_main_exist)