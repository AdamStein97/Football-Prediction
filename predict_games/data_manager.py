class DataManager():
    def __init__(self, player_database, heat_map_dict, player_attribute_features=("passing", "shooting", "physical", "tackling")):
        self.player_database = player_database
        self.heat_map_dict = heat_map_dict
        self.player_attribute_features = player_attribute_features

    #TODO: Implement
    def get_player_attributes(self, player_name):
        return 0

    def get_player_heat_map(self, position):
        return self.heat_map_dict[position]