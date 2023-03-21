import os
from dotenv import load_dotenv
from KZ_project.binance_domain.binance_client import BinanceClient
from KZ_project.logger.logger import Logger

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret_key = os.getenv('BINANCE_SECRET_KEY')



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
    LOG_PATH = './src/KZ_project/logger' + os.sep + "logs"
    LOG_FILE_NAME_PREFIX = f"log_{SYMBOL_CUT}_{start_date}_{end_date}"

    logger = Logger(LOG_PATH, LOG_FILE_NAME_PREFIX)

class BinanceConfig:
    SYMBOL = 'BTCUSDT'
    SYMBOL_NAME = 'Bitcoin'
    SYMBOL_CUT = 'btc'
    SCALE = 1
    range_list = [i for i in range(5,21)]
    range_list = [i*1 for i in range_list]
    interval = '1d'
    start_date = '2018-06-30'
    end_date = '2023-03-20'
    #start_date = '2022-06-17'
    #end_date = '2023-02-17'
    source = 'binance'
    LOG_PATH = './src/KZ_project/logger' + os.sep + "logs"
    LOG_FILE_NAME_PREFIX = f"log_{SYMBOL_CUT}_{start_date}_{end_date}"
    
    logger = Logger(LOG_PATH, LOG_FILE_NAME_PREFIX)
    client = BinanceClient(api_key, api_secret_key, logger=logger)

