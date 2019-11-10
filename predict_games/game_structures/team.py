from predict_games.game_structures.player import Player
import numpy as np

class Team():
    def __init__(self, data_manager, player_position_tuples, formation):
        self.players = []
        self.formation = formation
        for player_name in player_position_tuples:
            player = Player(player_name, data_manager)
            self.players.append(player)

        self.gen_team_attribute_matrix()


    def gen_team_attribute_matrix(self):
        matrix = self.players[0].attribute_matrix()
        for player in self.players[1:]:
            matrix = np.add(matrix, player.attribute_matrix())

        self.team_attribute_matrix = matrix