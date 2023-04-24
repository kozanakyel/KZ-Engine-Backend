# KZ Forecasting Engine with Sentiment Analysis and Decision Trees
The objective of this project is to use various techniques such as Decision Trees, Sentiment Analysis with Natural Language Processing, and Japanese Candlestick Art to forecast the structure of the next candle for popular cryptocurrencies like Bitcoin, Ethereum, Binance coin, Ripple, and Doge coin. The predictions will be made using a binary format and will aim to determine whether the price movement of the cryptocurrencies in the market will increase or decrease within the next 1-2 hours based on the candle structure.

In addition to these techniques, the research also involves studying other technical analysis and indicator structures, as well as incorporating Twitter data and sentiment analysis scores into the model.
The research resulted in the development of an application with a backend data and AI modelling pipeline, as well as a frontend for users to view the coin signal results and how many tweets were evaluated.

However, it is important to note that predicting the movement of cryptocurrencies can be challenging due to their volatility and the many external factors that may impact their value. Therefore, while this project may provide useful insights, investors should exercise caution and consider all relevant information before making investment decisions.

The motivation behind this project is that when people decide to buy a coin, generally the price of that coin in the stock market starts to fall, and when all indicators say "sell," the price goes up, or when they say "buy," the price goes down. This is called the point of financial saturation. My goal is to create an artificial intelligence-based system that can prevent this from happening, by detecting these two points and presenting them to users through various methods. Additionally, I aim to turn this system into an application and meet with customers. By evaluating this process with artificial intelligence and applications, I want to create an application that will help people make long-term profits and reduce trading stress by detecting price trends.

Financial saturation is a well-known phenomenon in the world of trading and investing, and it can be difficult for individuals to navigate. With the advancements in technology and the increasing popularity of artificial intelligence, there is an opportunity to create a system that can help individuals make better decisions when it comes to trading. By detecting the points of financial saturation, individuals can avoid buying or selling at the wrong time and potentially lose money. This project aims to develop such a system using various methods and algorithms, including the use of machine learning and neural networks. Additionally, by creating an application that can be used by customers, this system can be accessible to a wider audience, ultimately helping more people make profitable trades and achieve their financial goals.

## How installed and Run
Require specific python version because Binance API not upgraded for other new python verison.
We recommend pyenv and use python 3.8

```
python==3.8.1
```

```
$ git clone https://github.com/kozanakyel/KZ-Forecasting-Engine-Backend
$ cd KZ-Forecasting-Engine-Backend
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -U pip
$ pip install -U requirements.txt
$ pip install -e src
$ python -m nltk.downloader stopwords

```

OK! At the now tou can ready for running application but before you should add .env file with this format.

```
TW_access_token=...
TW_access_token_secret=...
TW_consumer_key=...
TW_consumer_secret=...
TW_BEARER_TOKEN=...

BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...

SECRET_KEY=...

PSQL_USER=postgres
PSQL_PWD=postgres
PSQL_HOST=localhost
PSQL_PORT=5433
PSQL_DB_NAME=kz_project

API_PORT=5000
API_HOST=127.0.0.1
```

Then we can ready for running application. Firstly you must start the web api project part.

```
$ python src/app.py
```

If ypu can track and getting instantly model prediction for 5 coins(BTCUSDT, BNBUSDT, ETHUSDT, XRPUSDT, DOGEUSDT). you must run this command:

```
$ cd src/Kz_project/ml_pipeline/services/binance_service
$ python ai_trader.py
```

## Explanation and some Result Comparison

You can see some backtest result in data/plots folder for this 5 coins

![DOGE coin 5 months backtest result for AI model](/data/plots/model_evaluation/doge/DOGEUSDT_binance_1h_model_backtest.png)

Below picture you can see model importance for this model and DOGE coin

![DOGE coin Feature importance for AI model](/data/plots/model_evaluation/doge/DOGEUSDT_binance_1h_model_importance.png)

## AI pipeline and KZ Engine Logic

![Project Logic Flow](/assets/images/KZ_project.jpg)

### some extra information

![Bitcoin Sentiment Analysis](/assets/images/btc_twitter_sentimen.png)

INNOVATION: Always ask myself why and when all the people say buy coin but then price decrease, Sell the coin then price UP.
So i innovate a new Indicator with combination 23 indicators and also some strategies most popular. then all this indicators converted to binary matrix. at the last step i sum up all rows for obtain some knowledge about them. I named it the KZ_INDEX/SCORE.
This indicator shows the all the indicators says strong buy signal and the bottom point say that all the indicators stromg sell. and you can see actually this 2 things works opposite direction. and you can define buy/sell points. but you cannot implement esaily this indicator because it has very complex calculation.
![KZ_INDEX/SCORE](/assets/images/kz_index.png)

Twitter and telegram APIs will be purchased for natural language process operations, and the related data will be provided with data mining and its APIs, and the effect of new features and people's thoughts and sentiments on this subject on the price will be monitored. With the process to be added to the data pipeline, the optimization and effects of the model will be observed. The results obtained in the last stage will be evaluated as both technical, deep learning and sentimental analysis, and it will be tried to determine where the price can go in the next candle. It seems like a topic that has been mentioned in many places, but when you enter it, it will be noticed how small the visible part of the iceberg is. My purpose in choosing this subject, which has been talked about so much and information pollution is at a high level. It comes from my curiosity to determine how price algorithms move in scientific ways.

Lets take a coffe;
USDT -> (TRC20): TQvoroCWUybkQTz7peXGMNMUe1BfjYSwMb