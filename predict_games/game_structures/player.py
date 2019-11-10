import numpy as np

class Player():
    def __init__(self, player_position_tuples, data_manager):
        self.name, self.position = player_position_tuples
        self.attributes = data_manager.get_player_attributes(self.name)
        self.position_heat_map = data_manager.get_player_heat_map(self.position)
        self.gen_attribute_matrix()

    def gen_attribute_matrix(self):
        matricies = []
        for attribute in self.attributes:
            matrix = attribute * self.position_heat_map
            matricies.append(matrix)

        self.attribute_matrix = np.matrix(matricies)
