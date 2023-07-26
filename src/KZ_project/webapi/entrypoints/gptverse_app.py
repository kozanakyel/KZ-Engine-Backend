from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
from datetime import datetime, date
from collections import defaultdict

from KZ_project.webapi.services.trading_advice_service import *

load_dotenv()

gpt_blueprint = Blueprint('gptverse', __name__)

daily_request_count = defaultdict(int)
daily_response_limit = 10

@gpt_blueprint.route('/ai_assistant', methods=['POST'])
def post_assistant_response():
    # Check if the daily limit is reached
    today = date.today().isoformat()
    if daily_request_count[today] >= daily_response_limit:
        return jsonify({'response': 'Today daily response limit is reached. Please try again tomorrow.'}), 403

    query = request.json['query']
    response = kayze_agent.get_response(query)    # kayze_agent created in service file

    # Increment the daily request count
    daily_request_count[today] += 1

    return jsonify({'response': response}), 201


@gpt_blueprint.route('/trading_advisor', methods=['POST'])
def post_trade_advice():
    symbol = request.json['symbol']
    print(symbol)
    openai = create_openai_model()
    fewshot = create_fewshot_template()
    df = get_ohlc_data(symbol)
    query_test = create_query(df, symbol)
    advice_test = get_response_llm(openai, fewshot, query_test)

    return jsonify({'response': f'For {symbol}: {advice_test[2:]}'}), 201
