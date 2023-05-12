FROM python:3.8.16-bullseye AS BASE

RUN apt-get update \
    && apt-get --assume-yes --no-install-recommends install \
        build-essential \
        curl \
        git
        
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvf ta-lib-0.4.0-src.tar.gz && \
    rm ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib

WORKDIR /kz_engine

RUN pip install --no-cache-dir --upgrade pip

COPY . .

RUN ["pip", "install", "-r", "requirements.txt"]
RUN ["pip", "install", "-e", "src/"]
RUN python -m nltk.downloader stopwords
