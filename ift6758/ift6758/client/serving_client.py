import json
import requests
import pandas as pd
import logging

sys.path.append('..')
from data.game_client import *


logger = logging.getLogger(__name__)

class ServingClient:
    def __init__(self, ip: str = "127.0.0.1", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        
        
        df_json = X.to_json()
        f"{self.base_url}/predict"
        
        
        result = request.post(f"{self.base_url}/predict",df_json )
        X['prediction'] = pd.read_json(result)
        return X
        
    def logs(self) -> dict:
        """Get server logs"""

        raise NotImplementedError("TODO: implement this function")

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        model_dict = {
            'workspace': workspace,
            'model': model,
            'version': version
        }
        response = requests.post(self.base_url + "/download_registry_model", json=model_dict)
        return response
    
    
    # Pour download un json et retourner un dataframe CLEAN
    @app.route("/game/game_id>", methods=["GET"])
    def get_game(game_id):
      game = Game(game_id)
      return game.clean
