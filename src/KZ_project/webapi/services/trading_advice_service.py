from datetime import date, timedelta

from dotenv import load_dotenv
import os
import pandas as pd
import pandas_ta as ta

from langchain import OpenAI
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate

from KZ_project.Infrastructure.services.binance_service.binance_client import BinanceClient

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key


def create_openai_model(
        model_name: str = 'text-curie-001',
        temperature: float = 0.5
):
    openai = OpenAI(
        # model_name='text-davinci-003',
        model_name=model_name,
        temperature=temperature
    )
    return openai


def create_fewshot_template():
    examples = [
        {
            "query": "RSI_10 daily value is 25 and EMA 7 daily value above than EMA 13 daily. What is your Advice for trading?",
            "answer": "The RSI value is 25 near the overselling if youe are holding any coin you think continue to hold this coin."
                      "otherwise you can wait for the RSI decreasing for good buying oppurtunity and "
                      "EMA 7 is greater than EMA 13 so that values says bullish trend is continueing. You can think about bullish position."
        }, {
            "query": "RSI 10 daily value is 85 and EMA 7 daily value under than EMA 13 daily."
                     "What is your Advice for trading?",
            "answer": "RSI value is 85 say this condition is overbought and you can sell any asset if you are hold any coin."
                      "Also EMA 13 is greater than EMA 7 that shows a coin is in Bearish condition and"
                      " you could sell if you are hold this coin."
        }
    ]

    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """The following are experts from conversations with an AI
    trading advisor assistant.
    The RSI and EMA indicator values are playing important role when trading on a daily basis.
     RSI data indicating a decline can indicate an overbought condition between 70-100. so you can sell your coin.
     Else it shows overselling condition between 0-25. so you can buy this coin.
     Overselling condition create a good oppurtunity for buyimg asset and overbought condition means that you should sell your asset.
     The EMA indicators generally says if lower daily period is
     above on higher daily period that means is bullish. if the condition is opposite for EMA than
     we can say that is bearish signal for selling assets.
    Here are some examples:
    """
    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

    return few_shot_prompt_template


def get_response_llm(model, fewshot_template, query: str):
    return model(fewshot_template.format(query=query))


def create_rsi_ema_query(
        rsi_daily,
        rsi_value,
        ema_above,
        ema_under
):
    return f"RSI_{rsi_daily} daily value is {rsi_value:.2f} and EMA_{ema_above} is above on EMA {ema_under} day. " \
           f"What is your advice for this RSI and EMA?"


def get_ohlc_data(symbol: str):
    symbol_list = ['bitcoin', 'ethereum', 'binance']
    if symbol not in symbol_list:
        raise KeyError(f'This symbol not in symbol list: {symbol}')
    dict_asset = {'bitcoin': 'BTCUSDT',
                  'ethereum': 'ETHUSDT',
                  'binance': 'BNBUSDT'}

    asset_symbol = dict_asset[symbol]

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret_key = os.getenv('BINANCE_SECRET_KEY')
    client = BinanceClient(api_key, api_secret_key)
    df = client.get_history(asset_symbol, '1d', (date.today() - timedelta(days=30)).isoformat())

    df['rsi_10'] = ta.rsi(df['close'], length=10)
    df['ema_7'] = ta.ema(df['close'], length=7)
    df['ema_13'] = ta.ema(df['close'], length=13)

    last_rsi = df['rsi_10'].iloc[-1]
    last_ema_7 = df['ema_7'].iloc[-1]
    last_ema_13 = df['ema_13'].iloc[-1]

    return last_rsi, last_ema_7, last_ema_13


if __name__ == '__main__':
    openai = create_openai_model()
    fewshot = create_fewshot_template()
    rsi_l, ema7_l, ema13_l = get_ohlc_data('binance')
    if ema7_l >= ema13_l:
        query_test = create_rsi_ema_query(10, rsi_l, 7, 13)
    else:
        query_test = create_rsi_ema_query(10, rsi_l, 13, 7)
    advice_test = get_response_llm(openai, fewshot, query_test)
    print(advice_test)
