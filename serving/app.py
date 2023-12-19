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
import pickle
from dotenv import load_dotenv
from waitress import serve

load_dotenv()

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

config = {
    "DEBUG": True,  # On veut déboguer (pour l'instant)
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 0
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app=app)

version_feature = {
    '1.1.0': ['goal_distance'],
    '1.2.0': ['angle'],
    '1.3.0': ['goal_distance', 'angle']
}

def load_model(workspace, model, version):
    output_path = f"models/{workspace}/{model}/{version}"

    model_file_path = os.path.join(output_path, 'model-data', 'comet-sklearn-model.pkl')

    if os.path.isfile(model_file_path):
        app.logger.info(f"Model is already loaded at {output_path}")
        response = {"Loading": "Success"}

    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        try:
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

    with open(os.path.join(output_path, 'model-data', 'comet-sklearn-model.pkl'), 'rb') as f:
        model = pickle.load(f)

    return model, response


# TODO: setup basic logging configuration
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

# TODO: any other initialization before the first request (e.g. load default model)
model, response = load_model('jhd', 'xgboost', '1.3.0')
cache.set("model", model)


@app.route("/hello")
def hello():
    return "Hello"


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # TODO: read the log file specified and return the data
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

    model, response = load_model(workspace, model, version)

    if response["Loading"] == 'Success':
        cache.set('model', model)
        cache.set('version', version)

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
    version = cache.get("version")
    app.logger.info("Read Json successfully")

    # TODO:
    # Read data and filter modelling features
    data = pd.DataFrame.from_dict(json)
    data_temp = data[version_feature[version]]

    predictions = model.predict_proba(data_temp)

    data_temp['event_team'] = data['event_team']
    data_temp['Model Output'] = predictions[:, 1]

    app.logger.info("Predictions made with success")


    return jsonify(data_temp.to_dict('list'))  # response must be json serializable!


if __name__ == '__main__':
    serve(app, port='8000')