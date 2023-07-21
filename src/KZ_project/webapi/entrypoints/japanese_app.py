from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
from KZ_project.Infrastructure.services.yahoo_service.yahoo_client import YahooClient
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator


MAIN_PATH = '/data/outputs/data_ind/'
PURE_PATH = '/data/pure_data/'
FEATURE_PATH = '/data/outputs/feature_data/'
PREFIX_PATH = '.'
SYMBOL = 'BTC-USD' 
PERIOD = "1y"
INTERVAL = '1d'
START_DATE = '2021-06-30'
END_DATE = '2022-07-01'
HASHTAG = 'btc'
scale = 1
range_list = [i for i in range(5,21)]
range_list = [i*scale for i in range_list]
source='yahoo'

y = YahooClient()



load_dotenv()

japanese_blueprint = Blueprint('japon', __name__)

@japanese_blueprint.route('/japanese', methods=['POST'])
def post_japanese_response():
    symbol = request.json['symbol']
    d = DataCreator(symbol=symbol, source=source, range_list=range_list, period=PERIOD, interval=INTERVAL, 
                start_date=START_DATE, end_date=END_DATE, 
                client=y) 
    japanese_df = d.get_candlesticks(is_complete=True)
    response = japanese_df.to_json()
    return response, 201