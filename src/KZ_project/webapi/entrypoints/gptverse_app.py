from flask import request, jsonify, Blueprint
from dotenv import load_dotenv

from KZ_project.webapi.services.trading_advice_service import *

load_dotenv()

gpt_blueprint = Blueprint('gptverse', __name__)

@gpt_blueprint.route('/ai_assistant', methods=['POST'])
def post_assistant_response():
    query = request.json['query']
    response = gptverse_agent.get_response(query)    # gptverse_agent created in servie file
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
