from abc import ABC, abstractmethod


class IDataSaver(ABC):
    
    @abstractmethod
    def save_data(self) -> None:
        pass
    
    @abstractmethod
    def remove_directory(self) -> None:
        pass