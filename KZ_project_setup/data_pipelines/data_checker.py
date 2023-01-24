from abc import ABC, abstractmethod

"""
@author: Ugur AKYEL
@date: 24/01/2023

Description: Base class for abstraction and create a Factory pattern
Chaeck the file, folder or database is exist any specific token or asets 
"""

class DataChecker(ABC):
    def __init__(self, symbol: str, main_path: str, pure_path: str, feature_path: str, prefix_path: str):
        self.symbol = symbol
        self.prefix_path = prefix_path
        self.main_path = main_path
        self.pure_path = pure_path
        self.feature_path = feature_path
    
    @abstractmethod
    def create_pure(self) -> str:
        pass
    
    @abstractmethod
    def create_main(self) -> str:
        pass
    
    @abstractmethod
    def create_feature(self) -> str:
        pass
    
    @property
    @abstractmethod
    def is_pure_exist(self):
        pass
    
    @property
    @abstractmethod
    def is_main_exist(self):
        pass
    
    @property
    @abstractmethod
    def is_feature_exist(self):
        pass 