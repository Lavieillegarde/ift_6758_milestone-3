"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
from flask_caching import Cache
import sklearn
import pandas as pd
import joblib
from comet_ml import API
import xgboost


import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

config = {
    "DEBUG": True,          # On veut déboguer (pour l'instant)
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 0
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app=app)

#TODO: ce hook cause un bogue pour l'instant

# @app.before_first_request
# def before_first_request():
#     """
#     Hook to handle any initialization before the first request (e.g. load model,
#     setup logging handler, etc.)
#     """
#     # TODO: setup basic logging configuration
#     logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
#
#     # TODO: any other initialization before the first request (e.g. load default model)
#     api = API(os.environ.get("COMET_API_KEY", None))
#     cache.set('api', api)
#     #TODO j'ai harcodé xgboost pour l'instant
#     model_path = 'serving/models/' + 'xgboost'
#     api.download_registry_model('jhd', 'xgboost', '1.0.0',
#                                 output_path="serving/models/")
#     xgb = xgboost.XGBClassifier()
#     xgb.load_model(model_path)
#     cache.set('model', xgb)


@app.route("/hello")
def hello():
    return "Hello"

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    raise NotImplementedError("TODO: implement this endpoint")

    response = None
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    api = API(os.environ.get("COMET_API_KEY", None))
    model_path = 'serving/models/' + "xgboost"
    api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="serving/models/")
    xgb = xgboost.XGBClassifier()
    xgb.load_model(model_path)
    cache.set('model', xgb)

    response = 'Success'

    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO:
    raise NotImplementedError("TODO: implement this enpdoint")
    
    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!
