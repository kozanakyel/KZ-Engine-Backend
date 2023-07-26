from KZ_project.Infrastructure.services.redis_chatbot_service.index_redis_service import IndexRedisService
from KZ_project.webapi.entrypoints.flask_app import app, kz_blueprint, db
from KZ_project.webapi.entrypoints.gptverse_app import gpt_blueprint
from KZ_project.webapi.entrypoints.japanese_app import japanese_blueprint

# gunicorn --chdir /mnt/c/Users/kozan/Desktop/Sen_Des_Proj/
# KZ-forecasting-engine-Backend/src/KZ_project/webapi --bind 0.0.0.0:5005 app:app

# export FLASK_APP=src/KZ_project/webapi/app.py
# flask run
# flask run --host=0.0.0.0 --port=5005


app.register_blueprint(kz_blueprint)
app.register_blueprint(gpt_blueprint)
app.register_blueprint(japanese_blueprint)

if __name__ == '__main__':
    redis_service = IndexRedisService()
    pdf_files = redis_service.get_pdf_files()
    redis_service.index_checker()
    redis_service.initiliaze_tokenizer()
    app.run(port=5005, host='0.0.0.0')
