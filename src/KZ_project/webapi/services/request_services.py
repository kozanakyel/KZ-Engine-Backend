from KZ_project.Infrastructure import config
import requests
import json

class RequestServices():
    
    @staticmethod
    def get_model_with_api(symbol, interval, ai_type):
        url = config.get_api_url()
        
        r = requests.get(
            f"{url}/forecast_model", json={
                "symbol": symbol,
                "interval": interval,
                "ai_type": ai_type
            }
        )   
        string_content = r.content.decode('utf-8')
        dict_content = json.loads(string_content) 
        return dict_content


"""    
    @staticmethod
    def post_crypto_with_api(name, ticker, description):
        url = config.get_api_url()
        
        r = requests.post(
            f"{url}/add_crypto", json={"name": name,
                                             "ticker": ticker,
                                             "description": description
                                             }
        )    
        return r.status_code
    
    @staticmethod
    def post_signaltracker_with_api(symbol, interval, ai_type, 
                                    signal, ticker, tweet_counts, 
                                    datetime_t):
        url = config.get_api_url()
        
        r = requests.post(
            f"{url}/add_signal_tracker", json={"symbol": symbol,
                                             "interval": interval,
                                             "ai_type": ai_type,
                                             "signal":signal,
                                             "ticker":ticker,
                                             "tweet_counts":tweet_counts,
                                             "datetime_t":datetime_t
                                             }
        )    
        return r.status_code
    
    @staticmethod
    def post_save_model_with_api(
        symbol, hashtag, source, feature_counts, 
        model_name, interval, ai_type, accuracy_score
    ):
        url = config.get_api_url()
        r = requests.post(
            f"{url}/add_forecast_model", json={"symbol": symbol,
                                             "interval": interval,
                                             "ai_type": ai_type,
                                             "hashtag":hashtag,
                                             "source":source,
                                             "feature_counts":feature_counts,
                                             "model_name":model_name,
                                             "accuracy_score":accuracy_score
                                             }
        )    
        return r.status_code  
        
"""