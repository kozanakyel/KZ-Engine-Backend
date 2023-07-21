from __future__ import annotations
from KZ_project.core.adapters.crypto_repository import CryptoRepository
from KZ_project.core.adapters.forecastmodel_repository import ForecastModelRepository
from KZ_project.core.adapters.signaltracker_repository import SignalTrackerRepository

from KZ_project.core.domain.forecast_model import ForecastModel
from KZ_project.core.domain.signal_tracker import SignalTracker

from KZ_project.core.domain.crypto import Crypto
from KZ_project.core.adapters.repository import AbstractBaseRepository
from KZ_project.core.domain.user import User


class InvalidName(Exception):
    pass


def add_user(
        wallet: str, username: str, email: str,
        repo: AbstractBaseRepository, session,
) -> None:
    # print(f'wallet list: {wallet} and type {type(username)}')
    user_list = repo.list()
    # print(f'wallet list: {user_list} and type {type(user_list)}')
    user_wallet_list = [x.wallet for x in user_list]
    # print(f'wallet list: {user_wallet_list} and type {type(user_wallet_list)}')

    if wallet in user_wallet_list:
        raise InvalidName(f'Error This Wallet is exist: {wallet}')
    repo.add(User(wallet, username, email))
    session.commit()


def get_user(
        wallet: str,
        repo: AbstractBaseRepository, session,
):
    user_list = repo.list()
    user_wallet_list = [x.wallet for x in user_list]

    if wallet not in user_wallet_list:
        raise InvalidName(f'Error This Wallet is not exist: {wallet}')
    result = repo.get(wallet)
    session.commit()
    return result


def add_crypto(
        name: str, ticker: str, description: str,
        repo: AbstractBaseRepository, session,
) -> None:
    crypto_list = repo.list()
    crypto_name_list = [x.name for x in crypto_list]

    if name in crypto_name_list:
        raise InvalidName(f'Error This name is exist: {name}')
    repo.add(Crypto(name, ticker, description))
    session.commit()


def get_crypto(
        ticker: str,
        repo: AbstractBaseRepository, session,
):
    crypto_list = repo.list()
    crypto_ticker_list = [x.ticker for x in crypto_list]

    if ticker not in crypto_ticker_list:
        raise InvalidName(f'Error This ticker is not exist: {ticker}')
    result = repo.get(ticker)
    session.commit()
    return result


def add_forecast_model(
        symbol: str, source: str, feature_counts: int, model_name: str,
        interval: str, ai_type: str, hashtag: str, accuracy_score: float,
        datetime_t: str, crypto, repo: AbstractBaseRepository, session
) -> None:
    # repo_cr = CryptoRepository(session)
    # finding_crypto = get_crypto(ticker=hashtag, repo=repo_cr, session=session)
    f = ForecastModel(symbol, source, feature_counts, model_name,
                      interval, ai_type, hashtag, accuracy_score, datetime_t, crypto)
    print(f"################## {f}")
    repo.add(ForecastModel(symbol, source, feature_counts, model_name,
                           interval, ai_type, hashtag, accuracy_score, datetime_t, crypto))
    session.commit()


def get_forecast_model(
        symbol: str, interval: str, ai_type: str,
        repo: AbstractBaseRepository, session,
):
    result = repo.get(symbol, interval, ai_type)

    session.commit()
    return result


def add_signal_tracker(
        signal: int, ticker: str, tweet_counts: int, japanese_candle:str, datetime_t: str, backtest_returns_data,
        forecast_model: ForecastModel,
        repo: AbstractBaseRepository, session
) -> None:
    repo.add(SignalTracker(signal, ticker, japanese_candle, 
                           tweet_counts, datetime_t, backtest_returns_data,
                           forecast_model))
    session.commit()


def get_signal_tracker(
        forecast_model_id: int,
        repo: AbstractBaseRepository, session,
):
    result = repo.get(forecast_model_id)

    session.commit()
    return result


def get_fm_models_list_all_unique_symbols(
        interval: str, ai_type: str,
        repo: AbstractBaseRepository, session,
):
    result = repo.get_last_forecast_models(interval, ai_type)

    session.commit()
    return result


def prediction_service_new_signaltracker(ai_type, Xt, next_candle_prediction,
                                         symbol, interval, hashtag, tweet_counts, japanese_candle, 
                                         backtest_returns_data, session):
    # session = get_session()
    repo = SignalTrackerRepository(session)
    repo_fm = ForecastModelRepository(session)
    try:
        # print(f'deneme forecast: {symbol}  {ai_type}')
        result_fm = get_forecast_model(
            symbol,
            interval,
            ai_type,
            repo_fm,
            session
        )
        add_signal_tracker(
            next_candle_prediction,
            hashtag,
            tweet_counts,
            japanese_candle,
            Xt,
            backtest_returns_data,
            result_fm,
            repo,
            session,
        )
    except (InvalidName) as e:
        return f'error occyred for signal tracker {symbol}'
    # print(f'Succes signaltracker for {symbol}')
    return f'Succes signaltracker for {symbol}'


def save_crypto_forecast_model_service(accuracy_score, session, ticker,
                                       symbol, source, feature_counts, model_name,
                                       interval, ai_type, datetime_t):
    # session = get_session()
    repo = ForecastModelRepository(session)
    repo_cr = CryptoRepository(session)
    try:
        finding_crypto = get_crypto(
            ticker=ticker,
            repo=repo_cr,
            session=session
        )

        add_forecast_model(
            symbol=symbol,
            source=source,
            feature_counts=feature_counts,
            model_name=model_name,
            interval=interval,
            ai_type=ai_type,
            hashtag=ticker,
            accuracy_score=accuracy_score,
            crypto=finding_crypto,
            datetime_t=datetime_t,
            repo=repo,
            session=session,
        )
    except (InvalidName) as e:
        return f'An errror for creating model {e}'
    # print(f'Succes creating save model for {symbol}')
    return f'Succesfully created model {symbol}'
