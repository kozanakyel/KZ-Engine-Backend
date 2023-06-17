from flask import Flask, request, jsonify, Blueprint
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import re
from KZ_project.Infrastructure.services.redis_chatbot_service.index_redis_service import IndexRedisService
from KZ_project.webapi.services.trading_advice_service import *

load_dotenv()

gpt_blueprint = Blueprint('gptverse', __name__)


@gpt_blueprint.route('/ai_project_assistant', methods=['POST'])
def post_assistant_response():
    f1_query = request.json['f1_query']
    redis_service = IndexRedisService()
    response_f1 = redis_service.response_f1_query(f1_query)
    return jsonify({'response': response_f1}), 201


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
