from sqlalchemy import text

from KZ_project.core.domain.crypto import Crypto

def test_saving_cryptos(session):
    cr = Crypto("bitcoin", "satoshi nakamoto")
    session.add(cr)
    session.commit()
    rows = session.execute(text(
        'SELECT name, description FROM "cryptos"'
    ))
    assert list(rows) == [('bitcoin', "satoshi nakamoto")]
    
def test_crypto_mapper_can_load(session):
    session.execute(text(
        "INSERT INTO cryptos (name, description) VALUES "
        '("bitcoin", "RED-CHAIR"),'
        '("ripple", "RED-TABLE"),'
        '("ethereum", "BLUE-LIPSTICK")'
    ))
    expected = [
        Crypto("bitcoin", "RED-CHAIR"),
        Crypto("ripple", "RED-TABLE"),
        Crypto("ethereum", "BLUE-LIPSTICK"),
    ]

    assert session.query(Crypto).all() == expected