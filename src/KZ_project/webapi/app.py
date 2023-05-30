from KZ_project.webapi.entrypoints.flask_app import app, kz_blueprint

app.register_blueprint(kz_blueprint)

if __name__ == '__main__':
    app.run()
