import pandas as pd
import os
import numpy as np
from PIL import Image

import predict_games as pg


position_dict = {
    "cb" : "Center Back",
    "rb" : "Right Back",
    "lb" : "Left Back",
    "cm" : "Center Midfield",
    "rm" : "Right Midfield",
    "lm" : "Left Midfield",
    "st" : "Striker",
    "rw" : "Right Wing",
    "lw" : "Left Wing",
    "fw" : "Center Forward",
    "am" : "Attacking Midfield"
}

class DataManager():
    def __init__(self, player_attribute_features=("Balance", "Shot_Power", "Short_Pass")):
        self.player_attribute_features = list(player_attribute_features)
        self.load_player_database()
        self.load_heat_map_dict()

    def load_heat_map_dict(self):
        self.heat_map_dict = {}
        for data in os.listdir(pg.HEATMAP_DIR):
            img = Image.open(os.path.join(pg.HEATMAP_DIR, data)).convert('L')
            pos_formation = data[:-4]
            self.heat_map_dict[pos_formation] = 255 - np.asarray(img)


    def load_player_database(self):
        player_database_filenames = [os.path.join(pg.PLAYER_DATA_DIR, file) for file in os.listdir(pg.PLAYER_DATA_DIR)]
        full_player_database = pd.read_csv(player_database_filenames[0])
        for file in player_database_filenames[1:]:
            database = pd.read_csv(file)
            full_player_database = full_player_database.append(database, ignore_index=True)
            full_player_database = full_player_database.drop_duplicates(subset=["Name"],keep='first')
        self.player_database = full_player_database

    def get_player_attributes(self, player_name):
        if not player_name in self.player_database["Name"].values:
            print(player_name)
        attributes = self.player_database[self.player_database["Name"] == player_name]
        attribute_values = attributes.iloc[0][self.player_attribute_features]
        return attribute_values

    def convert_pos_formation_to_filename_convention(self, position, formation):
        preprocessed_formation = formation.replace("-", "")
        preprocessed_position = position_dict[position[:2]]
        if preprocessed_position == "Center Back" and preprocessed_formation[0] == "4":
            preprocessed_position = "Left Center Back"
        if preprocessed_position == "Center Midfield" and preprocessed_formation[1] == "4":
            preprocessed_position = "Left Center Midfield"
        if preprocessed_position == "Center Forward" and preprocessed_formation[-1] == "2":
            preprocessed_position = "Left Center Forward"
        filename = "{}_{}".format(preprocessed_position, preprocessed_formation)
        return filename

    def get_player_heat_map(self, position, formation):
        key = self.convert_pos_formation_to_filename_convention(position, formation)
        return self.heat_map_dict[key]