import os
import predict_games as pg
import json
import pandas as pd
import math

from predict_games.data_manager import DataManager
from predict_games.batch_manager import BatchManager

class GamePredictModel():
    def __init__(self, heat_map_data_file_name="heat_map.json",player_data_file_name="player_data.csv",
                 match_data_file_name="match_data.json", config_name="config.json", train_perc=0.8):

        config = json.load(os.path.join(pg.CONFIG_DIR, config_name))
        self.batch_size = config["batch_size"]

        player_data = pd.read_csv(os.path.join(pg.DATA_DIR, player_data_file_name))
        match_data = pd.read_csv(os.path.join(pg.DATA_DIR, match_data_file_name))
        heat_map_data = json.load(os.path.join(pg.DATA_DIR, heat_map_data_file_name))
        self.data_manager = DataManager(player_data, heat_map_data)
        self.init_batch_managers(match_data, train_perc)


    #TODO: Should shuffle data otherwise test set will be only recent games
    def init_batch_managers(self, match_data, train_perc):
        test_perc = (1.0 - train_perc)/2
        train_set_size = int(math.floor(len(match_data) * train_perc))
        test_set_size =  int(math.floor(len(match_data) * test_perc))

        train_set = match_data[:train_set_size]
        validation_set = match_data[train_set_size:train_set_size + test_set_size]
        test_set = match_data[train_set_size + test_set_size:]

        self.train_batch_manager = BatchManager(self.batch_size)
        self.train_batch_manager.make_batches(train_set, self.data_manager)

        self.test_batch_manager = BatchManager(self.batch_size)
        self.test_batch_manager.make_batches(test_set, self.data_manager)

        self.val_batch_manager = BatchManager(self.batch_size)
        self.val_batch_manager.make_batches(validation_set, self.data_manager)