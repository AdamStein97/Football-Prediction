import math
import tensorflow as tf
import numpy as np

from predict_games.utils import score_to_ohv, shuffle_two_lists
from predict_games.game_structures.match import Match

class BatchManager():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.index = 0

    def gather_json_data(self, match_data,data_manager):
        x = []
        y = []
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
            except:
                print("Failed match")



        return x, y

    def make_batches(self, match_data, data_manager, test_perc=0.2):
        x_data, y_data = self.gather_json_data(match_data, data_manager)

        test_set_size = int(math.floor(len(x_data)/self.batch_size * test_perc))

        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(self.batch_size).shuffle(buffer_size=1000)

        self.test_batch_dataset = dataset.take(test_set_size)
        self.train_batch_dataset = dataset.skip(test_set_size)