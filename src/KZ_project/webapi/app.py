from KZ_project.webapi.entrypoints.flask_app import app, kz_blueprint

# gunicorn --chdir /mnt/c/Users/kozan/Desktop/Sen_Des_Proj/
# KZ-forecasting-engine-Backend/src/KZ_project/webapi --bind 127.0.0.1:5005 app:app

app.register_blueprint(kz_blueprint)

if __name__ == '__main__':
    app.run(port=5005, host='0.0.0.0')
