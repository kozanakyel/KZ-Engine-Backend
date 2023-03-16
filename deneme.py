import os
from dotenv import load_dotenv
from KZ_project.binance_domain.binance_client import BinanceClient

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret_key = os.getenv('BINANCE_SECRET_KEY')

client = BinanceClient(api_key, api_secret_key)