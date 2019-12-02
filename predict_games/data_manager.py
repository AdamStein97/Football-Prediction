import pandas as pd
import os, numpy, PIL
from PIL import Image

class DataManager():
    def __init__(self, player_database_filenames, player_attribute_features=("passing", "shooting", "physical", "tackling")):
        self.player_database_filenames = player_database_filenames
        self.player_attribute_features = player_attribute_features

    def load_heat_map_dict(self):
        self.heat_map_dict = {}
        for data in os.listdir('Average_Heatmaps/'):
            img = Image.open('Average_Heatmaps/' + data)
            pos_formation = data[:-4]
            self.heat_map_dict[pos_formation] = img


    def load_player_database(self):
        full_player_database = pd.read_csv(self.player_database_filenames[0])
        for file in self.player_database_filenames[1:]:
            database = pd.read_csv(file)
            filtered = database[database["Name"] not in full_player_database["Name"]]
            full_player_database.append(filtered, ignore_index=True)
        self.player_database = full_player_database

    def get_player_attributes(self, player_name):
        return self.player_database[self.player_database["Name"] == player_name][self.player_attribute_features]

    def convert_pos_formation_to_filename_convention(self, position, formation):
        return 0

    def get_player_heat_map(self, position, formation):
        key = self.convert_pos_formation_to_filename_convention(position, formation)
        return self.heat_map_dict[position]