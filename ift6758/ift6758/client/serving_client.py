import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "127.0.0.1", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = [
                'period', 'goal_distance'
            ]
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

        logger.info("Pinging Game")
        if X is not None:
            X_dict = {}
            X_values = X.values.tolist()
            X_dict['values'] = X_values
            pred = requests.post(url=self.base_url + "/predict", json=X_dict)
            try:
                output = pred.json()
                logger.info('DataFrame downloaded')  # length: ' + str(len(output)))
                return pd.DataFrame(output)
            except Exception as e:
                logger.info("Error in prediction")
                logger.info(str(e))
                return pd.DataFrame([0])

        logger.info("Tried to do prediction on None input.")
        logger.info("Returned output is None.")
        return None

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
