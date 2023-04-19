import os
from dotenv import load_dotenv
from KZ_project.ml_pipeline.services.binance_service.binance_client import BinanceClient
from KZ_project.Infrastructure.logger.logger import Logger
from datetime import datetime

load_dotenv()

api_key = os.getenv('BINANCE_API_KEY')
api_secret_key = os.getenv('BINANCE_SECRET_KEY')


LOG_PATH = './src/KZ_project/Infrastructure/logger' + os.sep + "logs"
LOG_FILE_NAME_PREFIX = f"log_{datetime.now()}"
logger = Logger(LOG_PATH, LOG_FILE_NAME_PREFIX)
client = BinanceClient(api_key, api_secret_key, logger=logger)

PSQL_USER = os.getenv('PSQL_USER')
PSQL_PWD = os.getenv('PSQL_PWD')
PSQL_HOST = os.getenv('PSQL_HOST')
PSQL_PORT = os.getenv('PSQL_PORT')
PSQL_DB_NAME = os.getenv('PSQL_DB_NAME')

API_PORT = os.getenv('API_PORT')
API_HOST = os.getenv('API_HOST')


class YahooConfig:
    SYMBOL = 'BTC-USD'
    SYMBOL_NAME = 'Bitcoin'
    SYMBOL_CUT = 'btc'
    SCALE = 1
    range_list = [i for i in range(5,21)]
    range_list = [i*1 for i in range_list]
    interval = '1h'
    #start_date = '2020-06-30'
    #end_date = '2022-07-01'
    start_date = '2022-06-17'
    end_date = '2023-02-17'
    source = 'yahoo'
    LOG_PATH = './src/KZ_project/Infrastructure/logger' + os.sep + "logs"
    LOG_FILE_NAME_PREFIX = f"log_{SYMBOL_CUT}_{start_date}_{end_date}"

    logger = logger
    


class BinanceConfig:
    SYMBOL = 'BNBUSDT'
    SYMBOL_NAME = 'Binance'
    SYMBOL_CUT = 'bnb'
    SCALE = 1
    description = 'CEO of Binance is CZ and big company in blockchain field.'
    range_list = [i for i in range(5,21)]
    range_list = [i*1 for i in range_list]
    interval_model = '1h'
    interval = '2h'
    start_date = '2022-06-17'
    end_date = '2023-03-22'
    #start_date = '2022-06-17'
    #end_date = '2023-02-17'
    source = 'binance'
    LOG_PATH = './src/KZ_project/Infrastructure/logger' + os.sep + "logs"
    LOG_FILE_NAME_PREFIX = f"log_{SYMBOL_CUT}_{start_date}_{end_date}"
    
    logger = logger
    client = client
    
class BitcoinConfig:
    SYMBOL = 'BTCUSDT'
    SYMBOL_NAME = 'Bitcoin'
    SYMBOL_CUT = 'btc'
    SCALE = 1
    description = 'Maded by Satoshi Nakamoto in 2008'
    range_list = [i for i in range(5,21)]
    range_list = [i*1 for i in range_list]
    interval_model = '1h'
    interval = '2h'
    start_date = '2022-06-17'
    end_date = '2023-03-22'
    #start_date = '2022-06-17'
    #end_date = '2023-02-17'
    source = 'binance'
    LOG_PATH = './src/KZ_project/Infrastructure/logger' + os.sep + "logs"
    LOG_FILE_NAME_PREFIX = f"log_{SYMBOL_CUT}_{start_date}_{end_date}"
    tweet_file_hourly = './data/tweets_data/btc/btc_hour.csv'
    tweet_file_daily = './data/tweets_data/btc/btc_day.csv'
    
    logger = logger
    client = client
    
class RippleConfig:
    SYMBOL = 'XRPUSDT'
    SYMBOL_NAME = 'Ripple'
    SYMBOL_CUT = 'xrp'
    SCALE = 1
    description = 'Big opputunity in Banking fields Future'
    range_list = [i for i in range(5,21)]
    range_list = [i*1 for i in range_list]
    interval_model = '1h'
    interval = '2h'
    start_date = '2022-06-17'
    end_date = '2023-03-22'
    #start_date = '2022-06-17'
    #end_date = '2023-02-17'
    source = 'binance'
    LOG_PATH = './src/KZ_project/Infrastructure/logger' + os.sep + "logs"
    LOG_FILE_NAME_PREFIX = f"log_{SYMBOL_CUT}_{start_date}_{end_date}"
    tweet_file_hourly = './data/tweets_data/xrp/xrp_hour.csv'
    tweet_file_daily = './data/tweets_data/xrp/xrp_day.csv'
    
    logger = logger
    client = client
    
class DogeConfig:
    SYMBOL = 'DOGEUSDT'
    SYMBOL_NAME = 'Doge coin'
    SYMBOL_CUT = 'doge'
    SCALE = 1
    description = 'What the fuck are you doing Elon??'
    range_list = [i for i in range(5,21)]
    range_list = [i*1 for i in range_list]
    interval_model = '1h'
    interval = '2h'
    start_date = '2022-06-17'
    end_date = '2023-03-22'
    #start_date = '2022-06-17'
    #end_date = '2023-02-17'
    source = 'binance'
    LOG_PATH = './src/KZ_project/Infrastructure/logger' + os.sep + "logs"
    LOG_FILE_NAME_PREFIX = f"log_{SYMBOL_CUT}_{start_date}_{end_date}"
    tweet_file_hourly = './data/tweets_data/doge/doge_hour.csv'
    tweet_file_daily = './data/tweets_data/doge/doge_day.csv'
    
    logger = logger
    client = client

class EthereumConfig:
    SYMBOL = 'ETHUSDT'
    SYMBOL_NAME = 'Etherum'
    SYMBOL_CUT = 'eth'
    SCALE = 1
    description = 'Ethereum is the modern and changeable bitcoin'
    range_list = [i for i in range(5,21)]
    range_list = [i*1 for i in range_list]
    interval_model = '1h'
    interval = '2h'
    start_date = '2022-06-17'
    end_date = '2023-03-22'
    #start_date = '2022-06-17'
    #end_date = '2023-02-17'
    source = 'binance'
    LOG_PATH = './src/KZ_project/Infrastructure/logger' + os.sep + "logs"
    LOG_FILE_NAME_PREFIX = f"log_{SYMBOL_CUT}_{start_date}_{end_date}"
    tweet_file_hourly = './data/tweets_data/doge/doge_hour.csv'
    tweet_file_daily = './data/tweets_data/doge/doge_day.csv'
    
    logger = logger
    client = client

config_coin_list = [BinanceConfig(), BitcoinConfig(), RippleConfig(), EthereumConfig(), DogeConfig()]


def get_postgres_uri():
    host = PSQL_HOST
    port = PSQL_PORT
    password = PSQL_PWD
    user, db_name = PSQL_USER, PSQL_DB_NAME
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def get_api_url():
    host = API_HOST
    port = API_PORT
    return f"http://{host}:{port}"
