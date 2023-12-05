import streamlit as st
import pandas as pd
import numpy as np
import os

from ift6758.client.serving_client import ServingClient


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

# """
# General template for your streamlit app.
# Feel free to experiment with layout and adding functionality!
# Just make sure that the required functionality is included as well
# """

st.title("Hockey Game Prediction Tool")
st.write("Base URL:", base_url)

with st.sidebar:
    workspace = st.selectbox(label='Workspace', options=['jhd'])
    model = st.selectbox(label='Model', options=model_version.keys())
    version = st.selectbox(label='Model version', options=model_version[model])

    model_button = st.button('Get model')

    if model_button:
        st.session_state['model'] = model
        st.session_state.servingClient.download_registry_model(workspace, st.session_state.model, version)
        st.write(f'Downloaded model:\n **{st.session_state.model}**')

with st.container():
    game_id = st.text_input(label='Game ID:', value='2016020001', max_chars=10)
    game_button = st.button('Ping game')

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass