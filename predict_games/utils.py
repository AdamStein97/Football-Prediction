import random

def score_to_ohv(score):
    return 0

def shuffle_two_lists(x, y):
    zipped = zip(x, y)

    random.shuffle(zipped)

    x, y = zip(*zipped)

    return x, y