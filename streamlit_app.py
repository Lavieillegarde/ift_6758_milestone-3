import streamlit as st
import pandas as pd
import os

from ift6758.client.serving_client import ServingClient
from ift6758.client.game_client import Game

IP = os.environ.get("SERVING_IP", "serving")
PORT = os.environ.get("SERVING_PORT", 5000)
base_url = f"http://127.0.0.1:{PORT}"

model_version = {
    'logisticregression': ['1.1.0', '1.2.0', '1.3.0']
}

changed_version = False

if 'servingClient' not in st.session_state:
    servingClient = ServingClient(ip=IP, port=PORT)
    st.session_state['servingClient'] = servingClient

if 'model_downloaded' not in st.session_state:
     st.session_state['model_downloaded'] = False

if 'game_id' not in st.session_state:
    st.session_state['game_id'] = ''

if 'clean_game' not in st.session_state:
    st.session_state['clean_game'] = None

if 'model_version' not in st.session_state:
    st.session_state['model_version'] = None


st.title("Hockey Visualization App")
st.write("Base URL:", base_url)

with st.sidebar:
    workspace = st.selectbox(label='Workspace', options=['jhd'])
    model = st.selectbox(label='Model', options=model_version.keys())
    version = st.selectbox(label='Model version', options=model_version[model])

    if version != st.session_state['model_version']:
        st.session_state['model_version'] = version
        st.session_state['clean_game'] = None
        st.session_state['game_id'] = ''

    model_button = st.button('Get model')

    if model_button:
        st.session_state['model'] = model
        st.session_state.servingClient.download_registry_model(workspace, st.session_state.model, version)
        st.session_state['model_downloaded'] = True
        st.write(f'Downloaded model:\n **{st.session_state.model}**')

with st.container():
    game_id = st.text_input(label='Game ID:', value='2023020001', max_chars=10)
    game_button = st.button('Ping game')


    if game_button:
        if not st.session_state['model_downloaded']:
            st.write('Please download model first!')
        else:

            old_game = False
            if game_id == st.session_state['game_id']:
                old_game = True


            st.session_state['game_id'] = game_id
            game = Game(game_id, old_game)


            if game.status:

                game.feat_eng_part2()

                clean_game = game.updated_clean_game

                st.session_state['model_version'] = version

                current_state = game.current_state
                if not clean_game.empty :

                    predictions = st.session_state.servingClient.predict(clean_game)
                    # On réarrange l'ordre des colonnes
                    cols = list(predictions.drop(columns=['Model Output', 'event_team']).columns.values)
                    cols.append('event_team')
                    cols.append('Model Output')

                    predictions = predictions[cols]

                    if st.session_state['clean_game'] is not None and old_game:
                        st.session_state['clean_game'] = pd.concat([st.session_state['clean_game'], predictions],
                                                                   ignore_index=True)
                        predictions = st.session_state['clean_game']
                    else:
                        st.session_state['clean_game'] = predictions

                # Le dataFrame est vide, ce qui veut dire que la game est terminée.
                # Nous ne changeons pas les prédictions dans ce cas
                else:
                    predictions = st.session_state['clean_game']


                xg_per_team = predictions.groupby('event_team')['Model Output'].sum().reset_index()
                xg_scores = dict(zip(xg_per_team['event_team'], xg_per_team['Model Output']))

                xg_home = round(xg_scores[current_state[0]['home']['teamName']], 2)
                delta_home = round(xg_home - current_state[3][current_state[0]['home']['teamName']], 2)
                xg_away = round(xg_scores[current_state[0]['away']['teamName']], 2)
                delta_away = round(xg_away - current_state[3][current_state[0]['away']['teamName']], 2)

                st.write(f" ## Game {game_id}: {current_state[0]['home']['teamName']} vs {current_state[0]['away']['teamName']} ")

                if not current_state[4]:
                    st.write(f"Period {str(current_state[1])} - {str(current_state[2])} left")
                else:
                    st.write("Game Finished")

                col1, col2 = st.columns([1, 1])

                col1.metric(label=f"{current_state[0]['home']['teamName']} xG (actual)",
                            value=f" {xg_home} ({current_state[3][current_state[0]['home']['teamName']]})", delta=delta_home)

                col2.metric(label=f"{current_state[0]['away']['teamName']} xG (actual)",
                            value=f"{xg_away} ({current_state[3][current_state[0]['away']['teamName']]})", delta=delta_away)

                st.write('\n \n \n #### Data used for predictions (with event team and predictions)')

                st.table(predictions)
            else:
                st.write("Invalid Game ID")

