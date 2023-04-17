
from KZ_project.core.domain.forecast_model import ForecastModel


class Crypto:
    def __init__(self, name:str, description):
        self.name = name.lower()
        self.description = description
        
    def __eq__(self, other):
        if not isinstance(other, Crypto):
            return False
        return other.name == self.name
        
        
    def __repr__(self):
        return f"<Crypto {self.name}>"