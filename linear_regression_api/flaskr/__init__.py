from flask import Flask
from flaskr.routes import lr_bp


def create_app():
    #Inicialización de la aplicación Flask
    app = Flask(__name__)

    #Registro de blueprints para las rutas
    app.register_blueprint(lr_bp)

    return  app