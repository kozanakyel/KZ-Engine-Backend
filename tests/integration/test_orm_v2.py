from sqlalchemy import text

from KZ_project.core.domain.crypto import Crypto
from KZ_project.core.domain.forecast_model import ForecastModel

def test_saving_cryptos(session):
    cr = Crypto("bitcoin", "btc","satoshi nakamoto")
    session.add(cr)
    session.commit()
    rows = session.execute(text(
        'SELECT name, ticker, description FROM "cryptos"'
    ))
    assert list(rows) == [('bitcoin', 'btc', "satoshi nakamoto")]
    
def test_crypto_mapper_can_load(session):
    session.execute(text(
        "INSERT INTO cryptos (name, ticker, description) VALUES "
        '("bitcoin", "btc","RED-CHAIR"),'
        '("ripple", "xrp","RED-TABLE"),'
        '("ethereum", "eth","BLUE-LIPSTICK")'
    ))
    expected = [
        Crypto("bitcoin", "btc","RED-CHAIR"),
        Crypto("ripple", "xrp","RED-TABLE"),
        Crypto("ethereum", "eth","BLUE-LIPSTICK"),
    ]

    assert session.query(Crypto).all() == expected
    
def test_saving_forecastmodel(session):
    cr = Crypto("bitcoin", "btc","satoshi nakamoto")
    fm = ForecastModel("BTCUSDT", "binance", 245, "modelname245", "1h", "xgboost", "btc", 34.456, cr.id)
    
    session.add(fm)
    session.commit()
    
    print(f'fm cryptos: {fm.crypto}')
    rows = list(session.execute(text('SELECT crypto_id, hashtag FROM "forecast_models"')))
    assert rows == [(cr.id, "btc")]
    
def test_retrieving_forecastmodel(session):
    cr = Crypto("bitcoin", "btc","satoshi nakamoto")
    print(cr.id)
    session.execute(
        text("INSERT INTO forecast_models "
             '(symbol, source, feature_counts, model_name, interval, ai_type, hashtag, accuracy_score, crypto_id) '
             'VALUES ("BTCUSDT", "binance", 245, "modelname245", "1h", "xgboost", "btc", 34.456, 2)'
             )
    )
    [[olid]] = session.execute(
        text("SELECT crypto_id FROM forecast_models WHERE symbol=:BTC AND ai_type=:type"),
        dict(BTC="BTCUSDT", type="xgboost"),
    )
    fm = session.query(ForecastModel).one()
    assert fm.crypto_id == olid