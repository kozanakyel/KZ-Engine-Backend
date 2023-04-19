
from KZ_project.core.domain.forecast_model import ForecastModel


class Crypto:
    def __init__(self, name:str, ticker:str, description:str):
        self.name = name.lower()
        self.ticker = ticker
        self.description = description
        
    def __eq__(self, other):
        if not isinstance(other, Crypto):
            return False
        return other.name == self.name
      
    def json(self):
        return {
            'name':self.name,
            'ticker':self.ticker,
            'description':self.description
            }  
        
    def __repr__(self):
        return f"<Crypto {self.name}>"