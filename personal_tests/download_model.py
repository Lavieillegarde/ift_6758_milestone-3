import pandas as pd
import comet_ml
import os
from comet_ml import API
import pickle
from dotenv import load_dotenv

load_dotenv()
api = API(os.environ["COMET_API_KEY"])

api.download_registry_model(
        workspace='jhd',
        registry_name='logisticregression',
        version='1.1.0',
        output_path="personal_tests/models",
        expand=True
    )

def load_model(workspace, model, version):

    output_path = f"serving/models/{workspace}/{model}/{version}"

    model_file_path = os.path.join(output_path, 'model-data', 'comet-sklearn-model.pkl')

    if os.path.isfile(model_file_path):
        response = {"Model_loading": "Success"}

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

            response = {"Model_loading": 'Success'}

        except:
            response = {"Model_loading": 'Failure'}
            model = None
            return model, response

    with open(os.path.join(output_path, 'model-data', 'comet-sklearn-model.pkl'), 'rb') as f:
        model = pickle.load(f)

    return model, response

model, response = load_model('jhd', 'xgboost', '1.5.0')