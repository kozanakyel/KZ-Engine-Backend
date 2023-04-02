from datetime import datetime
from flask import Flask, request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy

from KZ_project.Infrastructure.orm_mapper import orm
from KZ_project.service_layer import services
import KZ_project.Infrastructure.config as config

from KZ_project.adapters.tracker_repository import TrackerRepository
from KZ_project.adapters.asset_repository import AssetRepository


orm.start_mappers()
get_session = sessionmaker(bind=create_engine(config.get_postgres_uri()))
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = config.get_postgres_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


@app.route("/add_asset", methods=["POST"])
def add_asset():
    session = get_session()
    repo = TrackerRepository(session)

    services.add_asset(
        request.json["symbol"],
        request.json["source"],
        repo,
        session,
    )
    return "OK", 201


@app.route("/allocate_tracker", methods=["POST"])
def allocate_tracker():
    session = get_session()
    repo = AssetRepository(session)
    try:
        assetref = services.allocate_tracker_service(
            request.json["symbol"],
            request.json["datetime_t"],
            request.json["position"],
            repo,
            session,
        )
    except (services.InvalidSymbol) as e:
        return {"message": str(e)}, 400

    return {"assetref": assetref}, 201


@app.route("/position", methods=["POST"])
def get_position():
    session = get_session()
    repo = TrackerRepository(session)
    try:
        trackref = services.get_position(
            request.json["symbol"],
            repo,
            session
        )
        
    except (services.InvalidSymbol) as e:
        return {"message": str(e)}, 400

    return {"trackref": trackref}, 201