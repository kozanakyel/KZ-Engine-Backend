import os
from data_pipelines.data_manipulation import DataManipulation

if __name__ == '__main__':
    MAIN_PATH = '/data/outputs/data_ind/'
    PURE_PATH = '/data/pure_data/'
    FEATURE_PATH = '/data/outputs/feature_data/'
    PREFIX_PATH = './KZ_project'
    SYMBOL = 'BTC-USD' 
    SOURCE= 'yahoo'
    SCALE = 1
    RANGE_LIST = [i for i in range(5,21)]
    PERIOD = None
    INTERVAL = '1h'
    START_DATE = '2020-06-30'
    END_DATE = '2022-07-01'
    
    
    d = DataManipulation(symbol=SYMBOL, source=SOURCE, range_list=RANGE_LIST, 
                         period=PERIOD, interval=INTERVAL, start_date=START_DATE, 
                         end_date=END_DATE, prefix_path=PREFIX_PATH, 
                         main_path=MAIN_PATH, pure_path=PURE_PATH,
                         feature_path=FEATURE_PATH, saved_to_csv=True)  
    
    print(d.df["wma_15"][-1])