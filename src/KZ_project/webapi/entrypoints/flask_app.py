from datetime import datetime
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

from KZ_project.Infrastructure.orm_mapper import orm
from KZ_project.webapi.services import services
import KZ_project.Infrastructure.config as config

from KZ_project.core.adapters.tracker_repository import TrackerRepository
from KZ_project.core.adapters.asset_repository import AssetRepository
from KZ_project.core.adapters.aimodel_repository import AIModelRepository


orm.start_mappers()
get_session = sessionmaker(bind=create_engine(config.get_postgres_uri()))
app = Flask(__name__)

CORS(app)   # CORs policy from local development problem

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
        response_data = {
           "symbol": assetref[0],
            "position": assetref[1]
        }
    except (services.InvalidSymbol) as e:
        return {"message": str(e)}, 400

    return jsonify(response_data), 201


@app.route("/position", methods=["POST"])
def get_position():
    session = get_session()
    repo = TrackerRepository(session)
    try:
        result = services.get_position(
            request.json["symbol"],
            repo,
            session
        )
        response_data = {
            "symbol": result.symbol,
            "position": result.position,
            "datetime_t": result.datetime_t,
            "created_at": result.created_at
        }
        
    except (services.InvalidSymbol) as e:
        return {"message": str(e)}, 400

    return jsonify(response_data), 201

@app.route("/aimodel", methods=["POST"])
def get_aimodel():
    session = get_session()
    repo = AIModelRepository(session)
    try:
        result = services.get_aimodel(
            request.json["symbol"],
            repo,
            session
        )
        
        response_data = {
            "symbol": result.symbol, 
            "source": result.source, 
            "feature_counts": result.feature_counts, 
            "model_name": result.model_name, 
            "ai_type": result.ai_type, 
            "hashtag": result.hashtag, 
            "accuracy_score": result.accuracy_score, 
            "created_at": result.created_at,
            "file_path": result.get_filepath
        }
        
    except (services.InvalidSymbol) as e:
        return {"message": str(e)}, 400

    return jsonify(response_data), 201
