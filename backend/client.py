import requests
import json

class ServingClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def predict(self, input_data):
        endpoint = f"{self.base_url}/predict"
        response = requests.post(endpoint, json=json.loads(input_data.to_json()))
        return response.json()

    def logs(self):
        endpoint = f"{self.base_url}/logs"
        response = requests.get(endpoint)
        return response.text

    def download_registry_models(self, model_name):
        endpoint = f"{self.base_url}/download_registry_model"
        payload = {'model_name': model_name}
        response = requests.post(endpoint, json=payload)
        return response.json()

class GameClient:
    def __init__(self):
        self.processed_events = set()

    def query_live_game(self, game_id):
        endpoint = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        response = requests.get(endpoint)
        if response.status_code == 200:
            game_data = response.json()
            new_events = self.filter_processed_events(game_data['events'])
            self.process_events(new_events)
            return new_events  # Return only new events for processing
        else:
            return []

    def filter_processed_events(self, events):
        new_events = []
        for event in events:
            if event['event_id'] not in self.processed_events:
                new_events.append(event)
        return new_events

    def process_events(self, events):
        for event in events:
            # Process the events into features here
            # Use these features to query prediction service for goal probabilities
            # Store goal probabilities or relevant data as needed
            self.processed_events.add(event['event_id'])

