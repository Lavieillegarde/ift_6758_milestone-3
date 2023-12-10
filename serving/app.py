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
import comet_ml
from comet_ml import API
import xgboost
import pickle


#import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

config = {
    "DEBUG": True,          # On veut déboguer (pour l'instant)
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 0
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app=app)

comet_ml.init()

project_name = "project-ds-ift6758-a23"
workspace_name = "jhd"

#TODO: ce hook cause un bogue pour l'instant
# app.before_first_request is deprecated
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

def load_model(workspace, model, version):

    output_path = f"models/{workspace}/{model}/{version}"

    model_file_path = os.path.join(output_path, 'model-data', 'comet-sklearn-model.pkl')

    # check to see if the model you are querying for is already downloaded
    if os.path.isfile(model_file_path):
        # if yes, load that model and write to the log about the model change.  
        app.logger.info(f"Model is already loaded at {output_path}")
        response = {"Loading": "Success"}

    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        try:
            # if no, try downloading the model: if it succeeds, load that model and write to the log
            # about the model change. If it fails, write to the log about the failure and keep the 
            # currently loaded model
            api = API(os.environ.get("COMET_API_KEY", None))
            api.download_registry_model(
                workspace=workspace,
                registry_name=model,
                version=version,
                output_path=output_path,
                expand=True
            )

            app.logger.info(f"Loaded model at {output_path}")
            response = {"Loading": 'Success'}

        except:
            response = {"Loading": 'Failure'}
            app.logger.info(f"Model failed to load")

            return None, response

    
    # load model
    with open(os.path.join(output_path, 'model-data', 'comet-sklearn-model.pkl'), 'rb') as f:
        model = pickle.load(f)

    return model, response

# setup basic logging configuration
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

# load default model
model, response = load_model('jhd', 'xgboost', '1.1.0')
cache.set("model", model)

@app.route("/hello")
def hello():
    return "Hello"

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # read the log file specified and return the data
    with open(LOG_FILE, "r") as f:
        logs = f.read()

    response = {
        "logs": logs
    }

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

    workspace = json.get("workspace", None)
    model = json.get("model", None)
    version = json.get("version", None)

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    model, response = load_model(workspace, model, version)
    cache.set('model', model)

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
    model = cache.get("model")
    app.logger.info("Read Json successfully")

    # Read data and filter modelling features
    data = pd.DataFrame.from_dict(json)

    predictions = model.predict_proba(data)

    response = {
        "predictions_0": predictions[:,0].tolist(),
        "predictions_1": predictions[:,1].tolist()
    }    
    
    return jsonify(response)  # response must be json serializable!
