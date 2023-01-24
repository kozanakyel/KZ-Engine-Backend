import os
from folder_data_checker import FolderDataChecker

class FileDataChecker(FolderDataChecker):
    def __init__(self, symbol: str, main_path: str, pure_path: str, feature_path: str, prefix_path: str,
                 period: str, interval: str, start_date: str, end_date: str):
        super().__init__(symbol=symbol, 
                         prefix_path=prefix_path,
                         main_path=main_path, 
                         pure_path=pure_path, 
                         feature_path=feature_path)
        self.period = period
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        
    def create_pure(self) -> str:
        if self.period != None:
            self.pure_file = f'{self.symbol}_{self.period}_{self.interval}.csv'
        else:
            self.pure_file = f'{self.symbol}_{self.start_date}_{self.end_date}_{self.interval}.csv'
        return os.path.join(super().create_pure(), self.pure_file)
    
    def create_main(self) -> str:
        if self.period != None:
            self.main_file = f'{self.symbol}_df_{self.period}_{self.interval}.csv'
        else:
            self.main_file = f'{self.symbol}_df_{self.start_date}_{self.end_date}_{self.interval}.csv'
        return os.path.join(super().create_main(), self.main_file)
    
    def create_feature(self) -> str:
        if self.period != None:
            self.feature_file = f'{self.symbol}_df_feature_{self.period}_{self.interval}.csv'
        else:
            self.feature_file = f'{self.symbol}_df_feature_{self.start_date}_{self.end_date}_{self.interval}.csv'
        return os.path.join(super().create_feature(), self.feature_file)
    
    @property
    def is_pure_exist(self):
        return os.path.exists(self.create_pure())
    
    @property
    def is_main_exist(self):
        return os.path.exists(self.create_main())
    
    @property
    def is_feature_exist(self):
        return os.path.exists(self.create_feature())
        

if __name__ == '__main__':
    MAIN_PATH = '/data/outputs/data_ind/'
    PURE_PATH = '/data/pure_data/'
    FEATURE_PATH = '/data/outputs/feature_data/'
    PREFIX_PATH = './KZ_project_setup'
    SYMBOL = 'BTC-USD' 
    PERIOD = '1y'
    INTERVAL = '1h'
    START_DATE = '2020-06-30'
    END_DATE = '2022-07-01'
    
    d = FileDataChecker(symbol=SYMBOL, period=None, interval=INTERVAL, 
                    start_date=START_DATE, end_date=END_DATE, prefix_path=PREFIX_PATH, 
                    main_path=MAIN_PATH, pure_path=PURE_PATH,
                    feature_path=FEATURE_PATH)  
    
    print(os.getcwd())
    print(d.create_main())
    print(d.is_pure_exist)