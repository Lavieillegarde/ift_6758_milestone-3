from os.path import exists, isfile, isdir
import pandas as pd
import numpy as np
import json
import os
import re
import requests 


class DataCleaning:
    """
    Une classe de fonction utile pour nettoyer les donnees d'une partie brute json.
    La classe Game permet de range par instance les parties comme des objets rendant
    pratique la manipulation.

    Il est necessaire de specifier un chemin vers les donnees brutes incluant les donnees
    de play type.
    Les types de "play" consideres par defaut sont ['shot', 'goal','blocked-shot','missed-shot','shot-on-goal']
    """
    
    considered_event_list = ['shot', 'goal','blocked-shot','missed-shot','shot-on-goal']
    
    @staticmethod
    def extract_play(game: dict, play_id) -> dict:
        """
        Retourne l'information importante du "play" play_id pour une partie "game" donnee
        sous la forme d'un dictionaire
        """

        play_id = int(play_id) if type(play_id) == str else play_id
        play = game['play-by-play']['plays'][play_id]
        
        data = {}
        event_name = play['typeDescKey']
        
        if event_name in DataCleaning.considered_event_list:
            
            details = play.get('details', np.nan)
            gamePk = game['play-by-play']['id']
            sortOrder = play['sortOrder']
            # The team behind the event
            event_team_id = details['eventOwnerTeamId'] if details!=np.nan else np.nan
            event_team = game.teams['home']['teamName'] if game.teams['home']['team_id'] == event_team_id else game.teams['away']['teamName']
            
            event_description = play['typeDescKey']
            
            home_team = game.teams['home']['triCode']
            away_team = game.teams['away']['triCode']
            
            shot_type = details.get('shotType', np.nan) if details!=np.nan else np.nan
            
            # about
            eventIdx = play['eventId']
            period = play.get('period', np.nan)

            periodDescriptor = play.get('periodDescriptor', np.nan)
            periodType = periodDescriptor.get('periodType', np.nan) if periodDescriptor!= np.nan else np.nan
            periodTimeRemaining = play.get('timeRemaining', np.nan)
            

            x = details.get('xCoord', np.nan)
            y = details.get('yCoord', np.nan)

            goal_distance = np.nan
            
            strength = np.nan
            emptyNet = np.nan
            
            
            # Convert situation_code and event_owner_team_id to strings for processing
            situation_code = play['situationCode']
            
            # Extracting relevant information from the situation_code
            goalie_away = int(situation_code[0]) == 1
            skaters_away = int(situation_code[1])
            skaters_home = int(situation_code[2])
            goalie_home = int(situation_code[3]) == 1

            
            if game.teams['home']['team_id'] == event_team_id:
                emptyNet = goalie_home
                strength = goalie_home + skaters_home
                
            else:
                emptyNet = goalie_away
                strength = goalie_away + skaters_away

            players = {roster['playerId']: f"{roster['firstName']['default']} {roster['lastName']['default']}"
                       for roster in game['play-by-play']['rosterSpots']}
            
            for k, v in details.items():
                if 'PlayerId' in k:
                    data[k] = f"{players[v]}|{k.replace('PlayerId', '')}"
                if 'goalieInNetId' in k:
                    data[k] = f"{players[v]}|{k.replace('goalieInNetId', 'goalie')}"
                    emptyNet = False
                    
                    
            if details['eventOwnerTeamId'] ==  game.teams['home']['team_id']:
                team_ind = 'home'
                side = play.get("homeTeamDefendingSide", np.nan) 
            else:
                team_ind = 'away'
                side = play.get("homeTeamDefendingSide", np.nan) 
                side = 'left' if side =='right' else 'right'
                 
            goal_distance = (x - 89) ** 2 + y ** 2 if side == 'left' else (x + 89) ** 2 + y ** 2
            goal_distance = np.round(np.sqrt(goal_distance), 3)
            
            data['gamePk'] = gamePk
            
            data['event_name'] = event_name
            data['event_description'] = event_description
            data['event_team'] = event_team
            data['event_team_id'] = event_team_id
            
            data['home_team'] = home_team
            data['home_team_conf'] = np.nan #home_team_conf

            data['away_team'] = away_team
            data['away_team_conf'] = np.nan #away_team_conf

            data['shot_type'] = shot_type

            data['eventIdx'] = eventIdx
            data['sortOrder'] = sortOrder
            data['period'] = period
            data['periodType'] = periodType
            data['periodTimeRemaining'] = periodTimeRemaining

            data['x'] = x
            data['y'] = y

            data['goal_distance'] = goal_distance
            data['strength'] = strength
            data['emptyNet'] = emptyNet
            data['side'] = side

        return data

   
class DataAcquisition:

    @staticmethod
    def game_id_format(season: int, game_type: int, game_number: int):
        """
        Information pour former l'identifiant de partie
        gameID = XXXXYYZZZZ
        XXXX   = season eg 2017 for 2017-18
        YY     = type of game (https://statsapi.web.nhl.com/api/v1/gameTypes)
        ZZZZ   = Game number
        """
        return "{:04d}{:02d}{:04d}".format(season, game_type, game_number)

    @staticmethod
    def game_id_reverse_format(game_id: str):
        """
        Retourne les informations contenus dans le game_id
        """
        game_id = re.sub(r'[^0-9]', '', game_id)
        season = int(game_id[:4])
        game_type = int(game_id[4:6])
        game_number = int(game_id[6:])
        return season, game_type, game_number

    @staticmethod
    def url_format(game_id) -> str:
        """
        Format de l'url pour avoir acces a une partie avec l'API de la LNH
        """
        #return f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/'
        link0 = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play'
        link1 = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/landing'
        link2 = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore'
        
        return link0,link1,link2       

    @staticmethod
    def file_path_format(path: str, file_name: str, extension='json') -> str:
        return os.path.join(path, f"{file_name}.{extension}")

    @staticmethod
    def is_downloaded(path: str, file_name: str) -> bool:
        """
        Verifie si la partie a deja ete telechargee
        """
        return os.path.exists(DataAcquisition.file_path_format(path, file_name))

    @staticmethod
    def download_from_url(url: str, path: str, file_name: str) -> str:
        """
        Telecharge le json associe au lien.
        """
        file_path = DataAcquisition.file_path_format(path, file_name)
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(requests.get(url).text)

        return file_path

   
class Game(dict):
    def __init__(self, game_id=None):
        """
        Definie une instance de partie de la LNH dont les attributs sont
        id
        file_path (si définie)
        season
        game_type
        game_number
        clean (un dataframe nettoyer des donnees)
        """

        
        if old_game:
            self.old_game = Game(game_id)
            
        # print(os.curdir)
        data_path = os.path.join('ift6758', 'ift6758', 'data', 'game')

        file_path0 = DataAcquisition.file_path_format(data_path, f'play-by-play_{game_id}')
        file_path1 = DataAcquisition.file_path_format(data_path, f'landing_{game_id}')
        file_path2 = DataAcquisition.file_path_format(data_path, f'boxscore_{game_id}')

        # Si la partie existe
        if (exists(file_path0) and exists(file_path1) and exists(file_path2)) and not old_game:
            file_path0 = DataAcquisition.file_path_format(data_path, f'play-by-play_{game_id}')
            file_path1 = DataAcquisition.file_path_format(data_path, f'landing_{game_id}')
            file_path2 = DataAcquisition.file_path_format(data_path, f'boxscore_{game_id}')
                    
        else:
            url0, url1, url2 = DataAcquisition.url_format(game_id)# play, landing, boxscore

            file_path0 = DataAcquisition.download_from_url(url0, data_path,f'play-by-play_{game_id}')
            file_path1 = DataAcquisition.download_from_url(url1, data_path,f'landing_{game_id}')
            file_path2 = DataAcquisition.download_from_url(url2, data_path, f'boxscore_{game_id}')

        self.game_number = game_id
        
        super().__init__({'landing':{}, 'boxscore':{}, 'play-by-play':{}})
                    
        with open(file_path0, 'r') as js_file:
            with open(file_path0, encoding='utf-8') as fh:
                self['play-by-play'] = json.load(fh)
            
        with open(file_path1, 'r') as js_file:
            with open(file_path1, encoding='utf-8') as fh:
                self['landing'] = json.load(fh)
            
        with open(file_path2, 'r') as js_file:
            with open(file_path2, encoding='utf-8') as fh:
                self['boxscore'] = json.load(fh)
                
        if not old_game:
            self.old_game = self

    def reset_clean(self):
        """
        Ecrase la version clean actuel par celle de base
        """
        if hasattr(obj, '_clean'):
            del self._clean
            
        # reconstruit la version de base
        self.clean


    @property
    def empty(self):
        """
        Retourne un booleen si la game est vide
        """
        if not hasattr(self, '_empty'):
            self._empty = len(self['play-by-play']['plays'])<=1
        return self._empty
        
    @property
    def clean(self):
        """
        Retourne un dataframe des donnees nettoyees
        """
        if not hasattr(self, '_clean'):
            data_list = []

            for play_id in range(len(self['play-by-play']['plays'])):
                info = DataCleaning.extract_play(self, play_id)

                if info != {}:
                    data_list.append(info)

            self._clean = pd.DataFrame(data_list)

        return self._clean

    @property
    def teams(self):
        """
        Retourne les informations sur les equipes sous forme de dictionnaire.
        """
        if not hasattr(self, '_teams'):
            team_home = self['play-by-play']['homeTeam']
            team_away = self['play-by-play']['awayTeam']
            
            self._teams = {'home': {}, 'away': {}}

            self._teams['home']['triCode'] = team_home['abbrev']
            self._teams['home']['teamName'] = team_home['name']['default']
            self._teams['home']['team_id'] = team_home['id']
            
            self._teams['away']['triCode'] = team_away['abbrev']
            self._teams['away']['teamName'] = team_away['name']['default']
            self._teams['away']['team_id'] = team_away['id']

        return self._teams

    @property
    def id(self):
        """Retourne le numero d'identification de la partie"""
        return DataAcquisition.game_id_format(self.season, self.game_type, self.game_number)
    

    # Engineering features
    def calculate_angle(self, positive = False):
        '''
        Cette fonction permet de calculer l'angle de tir
        
        :param df: Dataframe (pandas df)
        :param positive: Angle de 0 à 180 si True sinon de -90 à 90 degrés
        :return: Caractéristique d'angle de tir (pandas Series)
        '''
        # Construit clean
        self.clean
        
        self._clean['angle_coord_x'] = np.where(self._clean['side'] == 'left', 90 - self._clean['x'], 90 + self._clean['x'])
        
        # Pour garder la symmétrie (tireur droitier vs gaucher),
        # par défaut les droitiers ont un angle plus petit que 90 degré
        self._clean['angle_coord_y'] = np.where(self._clean['side'] == 'left', -self._clean['y'], self._clean['y'])
        self._clean['angle'] = np.rad2deg(np.arctan(self._clean['angle_coord_x'] / self._clean['angle_coord_y']))
        
        self._clean = self._clean.drop(['angle_coord_x', 'angle_coord_y'], axis=1)
        
        if positive==True:
            self._clean['angle'] = np.where(self._clean['angle'] < 0, 180 + self._clean['angle'], self._clean['angle'])
        return self.clean
    
    def calculate_time_since_last(self):
        '''
        Cette fonction calcule le temps depuis le dernier évènement.
        
        :param df: Le dataframe (pandas df)
        :return: Caractéristique du temps depuis le dernier évènement (pandas Series)
        '''
        # Construit clean
        self.clean
        self._clean['period_diff'] = (self._clean['period'] - self._clean['period'].shift(1))
        self._clean['period_sec_last'] = self._clean['period_sec'].shift(1)

        # On calcule trois nouvelles colonnes intermédiaires pour construire notre colonne finale.
        # Soit le temps écoulé depuis le début de la période,
        self._clean['time_since_beg_period'] = 1200 - self._clean['period_sec']
        # Le temps des périodes pleines pour deux évènements consécutifs
        self._clean['time_full_period_diff'] = self._clean.apply(lambda x: (x.period_diff - 1) * 1200 if x.period_diff > 1 else 0, axis = 1)
        # La différence de temps lorsqu'on assume que les évènements sont dans la même période
        self._clean['time_diff_within_period'] = self._clean['period_sec_last'] - self._clean['period_sec']

        # Caractéristique finale
        self._clean['time_since_last'] = self._clean.apply(lambda x: x.period_sec_last + x.time_since_beg_period +\
                                                   x.time_full_period_diff if x.period_diff != 0\
                                                   else x.time_diff_within_period, axis = 1)
        return self.clean
    
    def feat_eng_part1(self, positive):
        # Construit clean
        self.clean
        
        self.calculate_angle(positive)

        self._clean['goal'] = self._clean['event_name'].apply(lambda x: 1 if x == 'goal' else 0)
        self._clean['emptyNet'] = self._clean['emptyNet'].apply(lambda x: 1 if x == True else 0)
        return self.clean

    def feat_eng_part2(self):
        '''
            Cette fonction accompli les étapes 1 et 2 d'ingénierie des caractéristiques.
    
            :param season: Saison (str)
            :param data_path: Path où trouver les données (str)
            :param save_path: Path où sauver les données (str)
            :return: Rien
        '''
        # Construit clean
        self.clean
        self._clean['regular_season'] = self._clean['gamePk'].astype(str).str[4:6] == '02'
        
        self.feat_eng_part1(positive=True)
        
        # Create all new features
        self._clean['period_sec'] = pd.to_timedelta('00:' + self._clean['periodTimeRemaining']).dt.total_seconds()
       
        self.calculate_time_since_last()
        
        
        # Restriction au type d'evenement
        self._clean = self._clean[self._clean['event_name'].isin(['shot', 'goal','blocked-shot','missed-shot','shot-on-goal'])]

        self._clean['last_event'] = self._clean['event_name'].shift(1)
        self._clean[['last_x', 'last_y']] = self._clean[['x', 'y']].shift(1)
        
        self._clean['last_distance'] = np.sqrt((self._clean['x'] - self._clean['last_x'])**2 + (self._clean['y'] - self._clean['last_y'])**2)
        
        # Seulement un rebond si le dernier évènement est un tir
        self._clean['rebound'] = self._clean['last_event'].apply(lambda x: True if x in ['shot','blocked-shot','missed-shot','shot-on-goal'] else False)

        
        self._clean['change_angle'] = np.abs(self._clean['angle'] - self._clean['angle'].shift(1))
        # Le changement d'angle est mis à 0 si ce n'est pas un rebond
        self._clean['change_angle'] =  np.where(self._clean['rebound'] == True, self._clean['change_angle'], 0)
        
        self._clean['speed'] = self._clean['last_distance']/self._clean['time_since_last']
        return self.clean
    
    @property 
    def current_scores(self):
        """
        Get number of goals from game
        """
        if not hasattr(self, '_scores'):
            dataframe = self.clean.copy()
            goals_per_team = dataframe.groupby('event_team')['goal'].sum().reset_index()
            self._scores = dict(zip(goals_per_team['event_team'], goals_per_team['goal']))
            
        return self._scores
        
    @property
    def current_state(self):
        teams = self.teams
        scores = self.current_scores
        
        last_event_order = self._clean['sortOrder'].max()
        last_event = self._clean[self._clean['sortOrder'] == last_event_order]
        
        period = last_event['period'].values
        time_remaning = last_event['periodTimeRemaining'].values
        return teams, scores, period, time_remaning
        
    @property
    def updated_clean_game(self):
        
        old_game_clean = self.old_game.clean
        old_game_clean.feat_eng_part2()

        game_clean = self.clean
        game_clean.feat_eng_part2()

        if id(self) == id(self.old_game):
            return old_game_clean
            
        return game_clean.merge(old_game_clean, indicator=True, how='left').loc[lambda x: x['_merge'] == 'left_only'].drop(columns='_merge')
        
                
        
