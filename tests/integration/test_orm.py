from KZ_project.core.domain.tracker import Tracker
from KZ_project.core.domain.asset import Asset
from KZ_project.core.domain.aimodel import AIModel
from datetime import date
from sqlalchemy import text

def test_saving_trackers(session):
    tr = Tracker("BTkUSDT", "2023-03-12 16:00:00+00:00", 1)
    session.add(tr)
    session.commit()
    rows = session.execute(text(
        'SELECT symbol, datetime_t, position FROM "trackers"'
    ))
    assert list(rows) == [("BTkUSDT", "2023-03-12 16:00:00+00:00", 1)]
    
    
def test_tracker_mapper_can_load(session):
    session.execute(text(
        "INSERT INTO trackers (symbol, datetime_t, position) VALUES "
        '("btcusdt", "RED-CHAIR", 12),'
        '("zecusd", "RED-TABLE", 13),'
        '("order2", "BLUE-LIPSTICK", 14)'
    ))
    expected = [
        Tracker("btcusdt", "RED-CHAIR", 12),
        Tracker("zecusd", "RED-TABLE", 13),
        Tracker("order2", "BLUE-LIPSTICK", 14),
    ]
    assert session.query(Tracker).all() == expected

def test_tracker_mapper_can_save(session):
    new_line = Tracker("order1", "DECORATIVE-WIDGET", 12)
    session.add(new_line)
    session.commit()

    rows = list(session.execute(text('SELECT symbol, datetime_t, position FROM "trackers"')))
    assert rows == [("order1", "DECORATIVE-WIDGET", 12)]  
    
def test_retrieving_assets(session):
    session.execute(text(
        "INSERT INTO assets (symbol, source)"
        ' VALUES ("batch1", "sku1")'
    ))
    session.execute(text(
        "INSERT INTO assets (symbol, source)"
        ' VALUES ("batch3", "sku2")'
    ))
    expected = [
        Asset("batch1", "sku1"),
        Asset("batch3", "sku2"),
    ]

    assert session.query(Asset).all() == expected


def test_saving_assets(session):
    asset = Asset("batch1", "sku1")
    session.add(asset)
    session.commit()
    rows = session.execute(text(
        'SELECT symbol, source FROM "assets"'
    ))
    assert list(rows) == [("batch1", "sku1")]
    
    
def test_saving_allocations_tracker(session):
    batch = Asset("batch1", "sku1")
    line = Tracker("batch1", "sku222", 10)
    batch.allocate_tracker(line)
    session.add(batch)
    session.commit()
    rows = list(session.execute(text('SELECT tracker_id, asset_id FROM "allocations_tracker"')))
    assert rows == [(batch.id, line.id)]
    
    
def test_retrieving_allocations_tarcker(session):
    session.execute(
        text('INSERT INTO trackers (symbol, datetime_t, position) VALUES ("order1", "sku1", 12)')
    )
    [[olid]] = session.execute(
        text("SELECT id FROM trackers WHERE symbol=:orderid AND datetime_t=:sku"),
        dict(orderid="order1", sku="sku1"),
    )
    session.execute(text(
        "INSERT INTO assets (symbol, source)"
        ' VALUES ("order1", "sku1111")'
    ))
    [[bid]] = session.execute(
       text("SELECT id FROM assets WHERE symbol=:ref AND source=:sku"),
        dict(ref="order1", sku="sku1111"),
    )
    session.execute(
        text("INSERT INTO allocations_tracker (tracker_id, asset_id) VALUES (:olid, :bid)"),
        dict(olid=olid, bid=bid),
    )

    batch = session.query(Asset).one()

    assert batch._allocations_tracker == {Tracker("order1", "sku1", 12)}
    
    
def test_aimodel_mapper_can_save(session):
    new_line = AIModel("order1", "DECORATIVE-WIDGET", 12, "bnb_2012_234", "xgboost", "bnb", 54.3)
    session.add(new_line)
    session.commit()

    rows = list(session.execute(text('SELECT symbol, source, feature_counts, model_name, ai_type, hashtag, accuracy_score FROM "aimodels"')))
    assert rows == [("order1", "DECORATIVE-WIDGET", 12, "bnb_2012_234", "xgboost", "bnb", 54.3)]  