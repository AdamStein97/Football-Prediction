from predict_games.utils import score_to_ohv
from predict_games.game_structures.match import Match

class BatchManager():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.index = 0

    def make_batches(self, match_data):
        batches_x = []
        batches_y = []

        for match in match_data:
            result_label = score_to_ohv(match['score'])
            #match_obj = Match()