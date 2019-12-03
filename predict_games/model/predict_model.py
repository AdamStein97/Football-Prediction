import os
import predict_games as pg
import json
import pandas as pd
import math
import random
import tensorflow as tf
import numpy as np

from predict_games.data_manager import DataManager
from predict_games.batch_manager import BatchManager
from predict_games.model.conv_network import ConvNetwork

class GamePredictModel():
    def __init__(self, match_data_file_name="match_data_epl_18.json", config_name="config.json", train_perc=0.8):
        np.random.seed(88)
        random.seed(8)
        tf.compat.v1.disable_eager_execution()


        config = json.load(open(os.path.join(pg.CONFIG_DIR, config_name)))
        self.batch_size = config["batch_size"]
        self.game_dim = config["game_dim"]
        self.player_attributes = config["player_attributes"]
        self.num_player_attributes = len(self.player_attributes)
        self.model_config = config["model_config"]

        self.epochs = config["epochs"]

        match_data = json.load(open(os.path.join(pg.MATCH_DATA_DIR, match_data_file_name)))
        self.data_manager = DataManager(player_attribute_features=self.player_attributes)
        self.init_batch_managers(match_data, train_perc)
        self.init_tensors()
        self.init_model()
        self.saver = tf.compat.v1.train.Saver()

    def init_tensors(self):
        self.match_matricies = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, self.num_player_attributes, self.game_dim[0], self.game_dim[1]))
        self.results = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 3))

    def init_model(self):
        self.model = ConvNetwork(self.model_config)
        self.y_pred, self.loss = self.model.call(self.match_matricies, self.results)
        self.opt = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

    def train(self):
        eval_freq = 100
        save_freq = 1000
        test_eval_batches = 20
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            average_loss = 0
            for i in range(self.epochs):
                batch_x, batch_y = self.train_batch_manager.get_next_batch()
                match_matrix = self.train_batch_manager.get_match_matrix(batch_x)
                loss, _ = sess.run([self.loss, self.opt], feed_dict={self.match_matricies: match_matrix, self.results : batch_y})
                average_loss += loss
                if i % eval_freq == 0 and i > 0:
                    print("Average Train Loss: {}".format(average_loss/eval_freq))
                    average_loss = 0
                    for j in range(test_eval_batches):
                        loss = sess.run(self.loss,
                                           feed_dict={self.match_matricies: match_matrix, self.results: batch_y})
                        average_loss += loss
                    print("Average Test Loss: {}".format(average_loss / test_eval_batches))
                    average_loss = 0
                if i % save_freq and i > 0:
                    self.saver.save(sess, 'model', global_step=i)


    def init_batch_managers(self, match_data, train_perc):
        random.shuffle(match_data)
        train_set_size = int(math.floor(len(match_data) * train_perc))

        train_set = match_data[:train_set_size]
        test_set = match_data[train_set_size:]

        self.train_batch_manager = BatchManager(self.batch_size)
        self.train_batch_manager.make_batches(train_set, self.data_manager)

        self.test_batch_manager = BatchManager(self.batch_size)
        self.test_batch_manager.make_batches(test_set, self.data_manager)

