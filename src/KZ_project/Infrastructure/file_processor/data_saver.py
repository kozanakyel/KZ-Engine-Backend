import os, shutil
import pandas as pd

from KZ_project.core.interfaces.Idata_saver import IDataSaver
    
class CSVDataSaver(IDataSaver):
    """ this class save pure, output ind, and features,
        get main, pure and featured path,
        MAybe you can use default path,
        if not exist the others ways
    """       
    
    def save_data(self, df: pd.DataFrame(), path: str, file: str) -> None:
        if not os.path.exists(os.path.join(path, file)):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, file), mode='a'): pass 
            df.to_csv(os.path.join(path, file))
            
    def remove_directory(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            print(f'The path is not exist. {path}')
            
class NoDataSaver(IDataSaver):
    
    def save_data(self, df: pd.DataFrame(), path: str, file: str) -> None:
        pass
            
    def remove_directory(self, path: str) -> None:
        pass

def factory_data_saver(saved_data: bool):
    if saved_data:
        data_saver = CSVDataSaver()
    else:
        data_saver = NoDataSaver()
    return data_saver
    
        
