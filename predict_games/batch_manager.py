import math
import tensorflow as tf
import numpy as np
import pickle
from sklearn import preprocessing
import predict_games as pg
import os

from predict_games.utils import score_to_ohv, shuffle_two_lists
from predict_games.game_structures.match import Match

class BatchManager():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.index = 0

    def gather_json_data(self, match_data,data_manager):
        x = []
        y = []
        success = 0
        failure = 0
        for match in match_data:
            result_label = score_to_ohv("{}-{}".format(match['home_score'], match['away_score']))
            home_player_position_tuples = match['home_lineup']
            away_player_position_tuples = match['away_lineup']
            home_formation = match['home_formation']
            away_formation = match['away_formation']
            try:
                match_obj = Match(data_manager, home_player_position_tuples, home_formation, away_player_position_tuples, away_formation)
                if np.isnan(match_obj.match_matrix).any():
                    print("NaN Found")
                if None in match_obj.match_matrix:
                    print("None Found")
                x.append(match_obj.match_matrix)
                y.append(result_label)
                success += 1
            except:
                print("Failed match")
                failure  += 1


        print("Succesful loaded : {}, Failed Load : {}, Perc loaded: {}".format(success, failure, success/(success + failure) * 100))

        # with open('formatted_x_data.json', 'w') as outfile:
        np.save(os.path.join(pg.FORMATTED_DATA_DIR, "x-"+self.filename),x)
        #with open('formatted_y_data.json', 'w') as outfile:
        np.save(os.path.join(pg.FORMATTED_DATA_DIR, "y-"+self.filename),y)

        mean = np.mean(np.array(x), axis=0)
        std = np.std(np.array(x), axis=0)

        x = [(a - mean)/std for a in x]

        test = []
        for x_sample in x:
            if x_sample.tostring() in test:
                print("Found duplicate")
            test.append(x_sample.tostring())

        return x, y

    def make_batches(self, match_data, data_manager, data_names, test_perc=0.2, load_data=False):
        self.filename = '-'.join(data_names)

        if load_data:
            x_data = np.load(os.path.join(pg.FORMATTED_DATA_DIR, "x-"+self.filename+".npy"))
            y_data = np.load(os.path.join(pg.FORMATTED_DATA_DIR, "y-"+self.filename+".npy"))
        else:
            x_data, y_data = self.gather_json_data(match_data, data_manager)

        test_set_size = int(math.floor(len(x_data)/self.batch_size * test_perc))

        print(test_set_size)

        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(self.batch_size).shuffle(buffer_size=1000)

        self.test_batch_dataset = dataset.take(test_set_size)
        self.train_batch_dataset = dataset.skip(test_set_size)