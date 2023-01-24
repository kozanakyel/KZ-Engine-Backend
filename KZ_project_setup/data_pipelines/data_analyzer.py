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
