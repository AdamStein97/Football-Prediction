from predict_games.model.predict_model import GamePredictModel
import os
from PIL import Image

for data in os.listdir('../data/heatmaps/'):
    img = Image.open('../data/heatmaps/'+data)
    img.thumbnail((30,22), Image.ANTIALIAS)
    img.save('../data/scaled_heatmaps/'+data, "png")

model = GamePredictModel()
model.train()