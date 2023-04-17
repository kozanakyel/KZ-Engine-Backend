
from KZ_project.core.domain.forecast_model import ForecastModel


class Crypto:
    def __init__(self, name:str):
        self.name = name
        self._forecast_models = []
        
    def add_forecast_model(self, forecast_model: ForecastModel):
        if forecast_model not in self._forecast_models:
            self._forecast_models.append(forecast_model)
            forecast_model._crypto = self
            
    def remove_forecast_model(self, forecast_model: ForecastModel):
        if forecast_model in self._forecast_models:
            self._forecast_models.remove(forecast_model)
            forecast_model._crypto = None    
    
        
    def __repr__(self):
        return f"<Crypto {self.name}>"