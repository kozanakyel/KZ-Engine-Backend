FROM python:3.8.16-bullseye AS BASE

RUN apt-get update \
    && apt-get --assume-yes --no-install-recommends install \
        build-essential \
        curl \
        git

WORKDIR /kz_engine

RUN pip install --no-cache-dir --upgrade pip

COPY . .

RUN ["pip", "install", "-r", "requirements.txt"]
RUN ["pip", "install", "-e", "src/"]
RUN python -m nltk.downloader stopwords

RUN python src/KZ_project/ml_pipeline/services/binance_service/ai_trader.py
RUN python src/app.py 
