import streamlit as st
import pandas as pd
import numpy as np
import os

from ift6758.client.serving_client import ServingClient
from ift6758.client.game_client import Game

IP = os.environ.get("SERVING_IP", "127.0.0.1")
PORT = os.environ.get("SERVING_PORT", 5000)
base_url = f"http://{IP}:{PORT}"

model_version = {
    'logisticregression': ['1.0.0', '1.1.0', '1.2.0', '1.3.0'],
    'xgboost': ['1.1.0', '1.2.0', '1.3.0', '1.4.0', '1.5.0', '1.6.0'],
    'ensemble-xgboost': ['1.1.0', '1.1.1'],
    'xgboost-k-best': ['1.1.0'],
    'randomforest': ['1.1.0']
}


if 'servingClient' not in st.session_state:
    servingClient = ServingClient(ip=IP, port=PORT)
    st.session_state['servingClient'] = servingClient

if 'model_downloaded' not in st.session_state:
     st.session_state['model_downloaded'] = False

# """
# General template for your streamlit app.
# Feel free to experiment with layout and adding functionality!
# Just make sure that the required functionality is included as well
# """

st.title("Hockey Visualization App")
st.write("Base URL:", base_url)

with st.sidebar:
    workspace = st.selectbox(label='Workspace', options=['jhd'])
    model = st.selectbox(label='Model', options=model_version.keys())
    version = st.selectbox(label='Model version', options=model_version[model])

    model_button = st.button('Get model')

    if model_button:
        st.session_state['model'] = model
        st.session_state.servingClient.download_registry_model(workspace, st.session_state.model, version)
        st.session_state['model_downloaded'] = True
        st.write(f'Downloaded model:\n **{st.session_state.model}**')

with st.container():
    game_id = st.text_input(label='Game ID:', value='2016020001', max_chars=10)
    game_button = st.button('Ping game')

    if game_button:
        st.session_state['game_id'] = game_id

        # raw_data_path = os.path.join('.', '..', 'data', 'raw')
        # processed_data_path = os.path.join('.', '..', 'data', 'processed')
        #
        # # Valider si le repertoir data/raw existe
        # if not os.path.exists(raw_data_path):
        #     os.makedirs(raw_data_path)
        #
        # # Valider si le repertoir data/procesed existe
        # if not os.path.exists(processed_data_path):
        #     os.makedirs(processed_data_path)

        # une fonction pour ranger le fichier a chemin
        # raw_data_path est inclu dans la classe DA

        if not st.session_state['model_downloaded']:
            st.write('Please download model first!]')
        else:
            game = Game(game_id)

            game.feat_eng_part2()

            clean_game = game.clean

            st.write(f'**The current game ID is {game_id}!**')


with st.container():
    if game_button and st.session_state['model_downloaded']:
        pass
with st.container():
    # TODO: Add data used for predictions
    pass