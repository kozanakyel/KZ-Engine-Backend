from sqlalchemy import Table, MetaData, Column, Integer, String, Date, ForeignKey, DateTime, Float
from sqlalchemy.orm import relationship, registry
from sqlalchemy.sql import func

from KZ_project.core.domain.tracker import Tracker
from KZ_project.core.domain.asset import Asset
from KZ_project.core.domain.aimodel import AIModel

mapper_registry = registry()

metadata = MetaData()

assets = Table(
    "assets",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(25)),
    Column("source", String(100))
    
)

aimodels = Table(
    "aimodels",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(25)),
    Column("source", String(100)),
    Column("feature_counts", Integer),
    Column("model_name", String(500)),
    Column("ai_type", String(200)),
    Column("hashtag", String(100), nullable=True),
    Column("accuracy_score", Float()),
    Column("created_at", DateTime())
)


trackers = Table(
    "trackers",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(25)),
    Column("datetime_t", String(200)),
    Column("position", Integer, nullable=False),
    Column("created_at", DateTime())
)

allocations_tracker = Table(
    "allocations_tracker",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tracker_id", ForeignKey("trackers.id")),
    Column("asset_id", ForeignKey("assets.id")),
)


def start_mappers():
    lines_mapper_tracker = mapper_registry.map_imperatively(
        Tracker, trackers
    )
    
    mapper_registry.map_imperatively(
        Asset, 
        assets,
         properties={
            "_allocations_tracker": relationship(
                lines_mapper_tracker, secondary=allocations_tracker, collection_class=set,
            )
        },
    )
    
    mapper_registry.map_imperatively(
        AIModel, aimodels
    )