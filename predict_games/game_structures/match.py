from predict_games.game_structures.team import Team

class Match():
    def __init__(self, data_manager, home_player_position_tuples, home_formation, away_player_position_tuples, away_formation):
        self.home_team = Team(data_manager, home_player_position_tuples, home_formation)
        self.away_team = Team(data_manager, away_player_position_tuples, away_formation)
        self.match_matrix = self.home_team.team_attribute_matrix - self.away_team.team_attribute_matrix
