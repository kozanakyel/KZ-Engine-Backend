from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
from KZ_project.Infrastructure.services.twitter_service.twitter_collection import (
    TwitterCollection,
)
from KZ_project.Infrastructure.services.yahoo_service.yahoo_client import YahooClient
from KZ_project.core.adapters.sentimentrecord_repository import (
    SentimentRecordRepository,
)
from KZ_project.ml_pipeline.data_pipeline.data_creator import DataCreator
from KZ_project.ml_pipeline.sentiment_analyzer.sentiment_analyzer import (
    SentimentAnalyzer,
)

from KZ_project.webapi.services import services
from KZ_project.webapi.entrypoints.flask_app import get_session


MAIN_PATH = "/data/outputs/data_ind/"
PURE_PATH = "/data/pure_data/"
FEATURE_PATH = "/data/outputs/feature_data/"
PREFIX_PATH = "."
SYMBOL = "BTC-USD"
PERIOD = "1mo"
INTERVAL = "1d"
START_DATE = "2021-06-30"
END_DATE = "2022-07-01"
HASHTAG = "btc"
scale = 1
range_list = [i for i in range(5, 21)]
range_list = [i * scale for i in range_list]
source = "yahoo"

y = YahooClient()
client = TwitterCollection()
sid = SentimentAnalyzer()

load_dotenv()

japanese_blueprint = Blueprint("japon", __name__)


@japanese_blueprint.route("/japanese", methods=["POST"])
def post_japanese_response():
    symbol = request.json["symbol"]
    d = DataCreator(
        symbol=symbol,
        source=source,
        range_list=range_list,
        period=PERIOD,
        interval=INTERVAL,
        start_date=START_DATE,
        end_date=END_DATE,
        client=y,
    )
    japanese_df = d.get_candlesticks(is_complete=True)
    response = japanese_df.to_json()
    return response, 201


@japanese_blueprint.route("/sentiment_analysis", methods=["POST"])
def post_sentiment_analysis():
    df = client.get_tweet_contents(tw_counts_points=100)
    df = sid.cleaning_tweet_data(df)
    df = sid.preprocessing_tweet_datetime(df)
    df = sid.get_sentiment_scores(df)
    sid.add_datetime_to_col(df)
    sent_scores = sid.get_sent_with_mean_interval(df, "1d")
    last_month = client.get_last_mont_df(sent_scores)
    response = last_month.to_json()
    return response, 201


@japanese_blueprint.route("/last_sentiment", methods=["POST"])
def post_last_sentiment():
    session = get_session()
    result = services.get_last_sentiment(session)
    return result.json(), 201


@japanese_blueprint.route("/all_sentiment", methods=["POST"])
def post_all_sentiment():
    session = get_session()
    result = services.get_all_sentiment_records(session)
    sentiment_records_json = [record.json() for record in result]
    return jsonify(sentiment_records_json), 201
