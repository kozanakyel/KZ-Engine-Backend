from flask import Flask, request, jsonify, Blueprint
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
# from flask_migrate import Migrate

from KZ_project.Infrastructure.orm_mapper import orm
from KZ_project.core.adapters.crypto_repository import CryptoRepository
from KZ_project.core.adapters.forecastmodel_repository import ForecastModelRepository
from KZ_project.core.adapters.signaltracker_repository import SignalTrackerRepository
from KZ_project.webapi.services import services
import KZ_project.Infrastructure.config as config

orm.start_mappers()
engine = create_engine(config.get_postgres_uri())
get_session = sessionmaker(bind=engine)
orm.metadata.create_all(engine)

app = Flask(__name__)

kz_blueprint = Blueprint('kz', __name__)


# CORs policy from local development problem
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = config.get_postgres_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


@kz_blueprint.route("/crypto", methods=["GET"])
def get_crypto():
    session = get_session()
    repo = CryptoRepository(session)
    try:
        result = services.get_crypto(
            request.json["name"],
            repo,
            session
        )

    except services.InvalidName as e:
        return {"message": str(e)}, 400

    return result.json(), 201


@kz_blueprint.route("/forecast_model", methods=["GET"])
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


@kz_blueprint.route("/signal_tracker", methods=["POST"])
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


@kz_blueprint.route("/signal_tracker_all", methods=["POST"])
def get_signal_tracker_all():
    """Fetch all unique models with symbol and last created date but 
    if we have models that same paramaters and 
    train different datetime you can see this models result

    Returns:
        _type_: _description_
    """
    session = get_session()
    repo_fm = ForecastModelRepository(session)
    repo_sg = SignalTrackerRepository(session)

    model_list = services.get_fm_models_list_all_unique_symbols(
        request.json["interval"],
        request.json["ai_type"],
        repo_fm,
        session
    )
    # print(model_list)
    signal_tracker_list = []
    for i in model_list:

        # print(i.id)
        res_signal = services.get_signal_tracker(
            i.id, repo_sg, session
        )
        # print(i.symbol)
        if res_signal:
            signal_tracker_list.append(res_signal.json())

    # json_obj_list = [i.json() for i in model_list]
    # print(json_obj_list)
    # json_output = json.dumps(json_obj_list)
    return signal_tracker_list, 201


"""
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
"""

"""
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
"""

"""
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
    
"""
