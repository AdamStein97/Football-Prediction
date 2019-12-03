import math

from predict_games.utils import score_to_ohv, shuffle_two_lists
from predict_games.game_structures.match import Match

class BatchManager():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.index = 0

    #TODO: Add betting odds fields
    def gather_json_data(self, match_data,data_manager):
        x = []
        y = []
        for match in match_data:
            result_label = score_to_ohv("{}-{}".format(match['home_score'], match['away_score']))
            home_player_position_tuples = match['home_lineup']
            away_player_position_tuples = match['home_lineup']
            home_formation = match['home_formation']
            away_formation = match['away_formation']
            try:
                match_obj = Match(data_manager, home_player_position_tuples, home_formation, away_player_position_tuples, away_formation)
                x.append(match_obj)
                y.append(result_label)
            except:
                print("Failed match")



        return x, y

    def make_batches(self, match_data, data_manager, shuffle=True):
        x_data, y_data = self.gather_json_data(match_data, data_manager)

        if shuffle:
            x_data, y_data = shuffle_two_lists(x_data, y_data)

        self.num_batches = int(math.floor(len(x_data) / self.batch_size))

        self.batches_x = []
        self.batches_y = []

        for i in range(self.num_batches):
            batch_x = []
            batch_y = []
            for j in range(self.batch_size):
                index = i * self.batch_size + j
                batch_x.append(x_data[index])
                batch_y.append(y_data[index])
            self.batches_x.append(batch_x)
            self.batches_y.append(batch_y)

    def get_next_batch(self):
        self.index = (self.index + 1) % self.num_batches
        return self.batches_x[self.index], self.batches_y[self.index]


    def get_match_matrix(self, batch_x):
        return [match.match_matrix for match in batch_x]
