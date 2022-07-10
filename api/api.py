from flask import Flask
from api.ml_router import ml_router

app = Flask(__name__)

app.register_blueprint(ml_router)
