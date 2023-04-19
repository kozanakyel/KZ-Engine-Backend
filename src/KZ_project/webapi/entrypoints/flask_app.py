from datetime import datetime
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate

from KZ_project.Infrastructure.orm_mapper import orm
from KZ_project.core.adapters.crypto_repository import CryptoRepository
from KZ_project.core.adapters.forecastmodel_repository import ForecastModelRepository
from KZ_project.core.adapters.signaltracker_repository import SignalTrackerRepository
from KZ_project.webapi.services import services
import KZ_project.Infrastructure.config as config

from KZ_project.core.adapters.tracker_repository import TrackerRepository
from KZ_project.core.adapters.asset_repository import AssetRepository
from KZ_project.core.adapters.aimodel_repository import AIModelRepository

##metadata = MetaData()

orm.start_mappers()
engine = create_engine(config.get_postgres_uri())
get_session = sessionmaker(bind=engine)
orm.metadata.create_all(engine)


app = Flask(__name__)

  # CORs policy from local development problem

app.config['SQLALCHEMY_DATABASE_URI'] = config.get_postgres_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
##migrate = Migrate(app, db)
####
with app.app_context():
    db.create_all()
CORS(app) 


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


###########################
@app.route("/add_crypto", methods=["POST"])
def add_crypto():
    session = get_session()
    repo = CryptoRepository(session)
    try: 
        services.add_crypto(
            request.json["name"],
            request.json["ticker"],
            request.json["description"],
            repo,
            session,
        )
    except (services.InvalidName) as e:
        return {"message": str(e)}, 400
    return "OK", 201

@app.route("/crypto", methods=["GET"])
def get_crypto():
    session = get_session()
    repo = CryptoRepository(session)
    try:
        result = services.get_crypto(
            request.json["name"],
            repo,
            session
        )
        
    except (services.InvalidName) as e:
        return {"message": str(e)}, 400

    return result.json(), 201

@app.route("/add_forecast_model", methods=["POST"])
def add_forecast_model():
    session = get_session()
    repo = ForecastModelRepository(session)
    repo_cr = CryptoRepository(session)
    try: 
        finding_crypto = services.get_crypto(
            ticker=request.json["hashtag"], 
            repo=repo_cr, 
            session=session)
    
        services.add_forecast_model(
            request.json["symbol"],
            request.json["source"],
            request.json["feature_counts"],
            request.json["model_name"],
            request.json["interval"],
            request.json["ai_type"],
            request.json["hashtag"],
            request.json["accuracy_score"],
            finding_crypto,
            repo,
            session,
        )
    except (services.InvalidName) as e:
        return {"message": str(e)}, 400
    return "OK", 201

@app.route("/forecast_model", methods=["GET"])
def get_forecast_model():
    session = get_session()
    repo = ForecastModelRepository(session)
    
    result = services.get_forecast_model(
            request.json["symbol"],
            request.json["interval"],
            request.json["ai_type"],
            repo,
            session
        )
    if result:
        return result.json(), 201
    else:
        return {"message": "Searching AI model is not found!"}, 400


@app.route("/add_signal_tracker", methods=["POST"])
def add_signal_tracker():
    session = get_session()
    repo = SignalTrackerRepository(session)
    repo_fm = ForecastModelRepository(session)
    try: 
        result_fm = services.get_forecast_model(
            request.json["symbol"],
            request.json["interval"],
            request.json["ai_type"],
            repo_fm,
            session
        )
        services.add_signal_tracker(
            request.json["signal"],
            request.json["ticker"],
            request.json["tweet_counts"],
            request.json["datetime_t"],
            result_fm,
            repo,
            session,
        )
    except (services.InvalidName) as e:
        return {"message": str(e)}, 400
    return "OK", 201

@app.route("/signal_tracker", methods=["GET"])
def get_signal_tracker():
    session = get_session()
    repo_fm = ForecastModelRepository(session)
    repo = SignalTrackerRepository(session)
    
    result_fm = services.get_forecast_model(
            request.json["symbol"],
            request.json["interval"],
            request.json["ai_type"],
            repo_fm,
            session
        )
    
    if result_fm:
        result = services.get_signal_tracker(
            result_fm.id,
            repo,
            session
        )
        if result:
            return result.json(), 201
        else:
            return {"message": "This Tracker is not found!"}, 400
    else:
        return {"message": "This AI model is not found!"}, 400