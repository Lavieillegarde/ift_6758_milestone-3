import sys
sys.path.append('ift_6758_milestone_3/ift6758')
from ift6758.client.serving_client import ServingClient

import requests
import json
import pandas as pd
import numpy as np
import os

Serving_client = ServingClient()
workspace = "jhd"
model = 'logisticregression'
version = '1.3.0'

features_list = ['goal_distance', 'angle']
loading_response = Serving_client.download_registry_model(workspace, model, version)
# output_path = f'./serving/models/{workspace}/{model}/{version}'
# with open(os.path.join(output_path, 'model-data', 'comet-sklearn-model.pkl'), 'rb') as f:
#     import pickle
#     model = pickle.load(f)

X_raw = pd.read_csv('ift6758/ift6758/data/wpg_v_wsh_2017021065.csv')

X = X_raw[X_raw['event_name'].isin(['Shot', 'Goal'])].reset_index(drop=True)
X = X[features_list]

# X_dict = json.loads(X.to_json())
# data = pd.DataFrame.from_dict(X_dict)
# predictions = model.predict_proba(data)
# response = {
#         "predictions": predictions
#     }
# pred = requests.post(url=f"http://127.0.0.1:5000" + "/predict", json=json.loads(X_dict)).json()

predictions = Serving_client.predict(X)