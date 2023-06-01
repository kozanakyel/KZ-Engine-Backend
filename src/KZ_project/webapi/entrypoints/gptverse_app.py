from flask import Flask, request, jsonify, Blueprint
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import re

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
