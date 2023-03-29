from KZ_project.domain.tracker import Tracker
from KZ_project.domain.asset import Asset

from KZ_project.adapters.asset_repository import AssetRepository
from KZ_project.adapters.tracker_repository import TrackerRepository

from sqlalchemy import text


def test_repository_can_save_a_asset(session):
    batch = Asset("batch1", "RUSTY-SOAPDISH")

    repo = AssetRepository(session)
    repo.add(batch)
    session.commit()

    rows = session.execute(
        text('SELECT symbol, source FROM "assets"')
    )
    assert list(rows) == [("batch1", "RUSTY-SOAPDISH")]


def insert_tracker(session):
    session.execute(text(
        "INSERT INTO trackers (symbol, datetime_t, position)"
        ' VALUES ("batch1", "GENERIC-SOFA", 12)')
    )
    [[orderline_id]] = session.execute(text(
        "SELECT id FROM trackers WHERE symbol=:orderid AND datetime_t=:sku"),
        dict(orderid="batch1", sku="GENERIC-SOFA"),
    )
    return orderline_id


def insert_asset(session, batch_id):
    session.execute(text(
        "INSERT INTO assets (symbol, source)"
        ' VALUES (:batch_id, "GENERIC-SOFA")'),
        dict(batch_id=batch_id),
    )
    [[batch_id]] = session.execute(text(
        'SELECT id FROM assets WHERE symbol=:batch_id AND source="GENERIC-SOFA"'),
        dict(batch_id=batch_id),
    )
    return batch_id


def insert_allocation_tracker(session, orderline_id, batch_id):
    session.execute(text(
        "INSERT INTO allocations_tracker (tracker_id, asset_id)"
        " VALUES (:orderline_id, :batch_id)"),
        dict(orderline_id=orderline_id, batch_id=batch_id),
    )


def test_repository_can_retrieve_a_asset_with_allocations(session):
    orderline_id = insert_tracker(session)
    batch1_id = insert_asset(session, "batch1")
    insert_asset(session, "batch2")
    insert_allocation_tracker(session, orderline_id, batch1_id)

    repo = AssetRepository(session)
    retrieved = repo.get("batch1")

    expected = Asset("batch1", "GENERIC-SOFA")
    assert retrieved == expected  # Batch.__eq__ only compares reference
    assert retrieved.symbol == expected.symbol
    assert retrieved.source == expected.source
    assert retrieved._allocations_tracker == {
        Tracker("batch1", "GENERIC-SOFA", 12),
    }