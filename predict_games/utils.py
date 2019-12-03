import random
import numpy as np

def score_to_ohv(score):
    split = score.split("-")
    home_goals = int(split[0])
    away_goals = int(split[-1])
    if home_goals > away_goals:
        return np.array([1, 0, 0])
    elif away_goals > home_goals:
        return np.array([0,0,1])
    else:
        return np.array([0,1,0])

def shuffle_two_lists(x, y):
    zipped = list(zip(x, y))

    random.shuffle(zipped)

    x, y = zip(*zipped)

    return x, y
