from KZ_project.webapi.entrypoints.flask_app import app, kz_blueprint
from KZ_project.webapi.entrypoints.gptverse_app import gpt_blueprint

# gunicorn --chdir /mnt/c/Users/kozan/Desktop/Sen_Des_Proj/
# KZ-forecasting-engine-Backend/src/KZ_project/webapi --bind 127.0.0.1:5005 app:app

# export FLASK_APP=src/KZ_project/webapi/app.py
# flask run
# flask run --host=0.0.0.0 --port=5000

app.register_blueprint(kz_blueprint)
app.register_blueprint(gpt_blueprint)

if __name__ == '__main__':
    app.run(port=5005, host='0.0.0.0')
