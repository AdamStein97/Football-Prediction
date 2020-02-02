import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "../data")
HEATMAP_DIR = os.path.join(DATA_DIR, "scaled_heatmaps")
MATCH_DATA_DIR = os.path.join(DATA_DIR, "match_data")
PLAYER_DATA_DIR = os.path.join(DATA_DIR, "player_data")
FORMATTED_DATA_DIR = os.path.join(DATA_DIR, "formatted_data")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

CONFIG_DIR = os.path.join(ROOT_DIR, "../configs")