from flask import Flask, request, jsonify, Blueprint
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import re

from KZ_project.webapi.services.trading_advice_service import *

load_dotenv()

gpt_blueprint = Blueprint('gptverse', __name__)

openai_api_key = os.getenv('OPENAI_API_KEY')
chatgpt = OpenAI(
    model_name='text-ada-001',
    temperature=0.3,
    max_tokens=150
)


@gpt_blueprint.route('/llm_response', methods=['POST'])
def post_question_to_llm():
    # Get the question from the request
    question = request.json['question']

    # Perform the OpenAI API call
    result = chatgpt(question)

    # Extract the numbers and newlines
    pattern = r'\n\d+\. '  # Matches newlines followed by numbers and a dot
    extract_n = re.sub(pattern, '', result)

    # Return the extracted response as JSON
    return jsonify({'response': extract_n[2:]})


@gpt_blueprint.route('/trading_advisor', methods=['POST'])
def post_trade_advice():
    symbol = request.json['symbol']
    try:
        rsi_l, ema7_l, ema13_l = get_ohlc_data(symbol)
    except KeyError as e:
        return {"message": str(e)}, 400
    openai = create_openai_model()
    fewshot = create_fewshot_template()

    if ema7_l >= ema13_l:
        query_test = create_rsi_ema_query(10, rsi_l, 7, 13)
    else:
        query_test = create_rsi_ema_query(10, rsi_l, 13, 7)
    advice_test = get_response_llm(openai, fewshot, query_test)

    return jsonify({'response': advice_test[2:]}), 201
