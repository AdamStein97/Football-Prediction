import os
import predict_games as pg
import json
import pandas as pd
import math
import random
import tensorflow as tf

from predict_games.data_manager import DataManager
from predict_games.batch_manager import BatchManager
from predict_games.model.conv_network import ConvNetwork

class GamePredictModel():
    def __init__(self, heat_map_data_file_name="heat_map.json",player_data_file_name="player_data.csv",
                 match_data_file_name="match_data.json", config_name="config.json", train_perc=0.8):

        config = json.load(os.path.join(pg.CONFIG_DIR, config_name))
        self.batch_size = config["batch_size"]
        self.game_dim = config["game_dim"]
        self.player_attributes = config["game_dim"]
        self.num_player_attributes = len(self.player_attributes)
        self.model_config = config["model_config"]

        self.epochs = config["epochs"]

        player_data = pd.read_csv(os.path.join(pg.DATA_DIR, player_data_file_name))
        match_data = pd.read_csv(os.path.join(pg.DATA_DIR, match_data_file_name))
        heat_map_data = json.load(os.path.join(pg.DATA_DIR, heat_map_data_file_name))
        self.data_manager = DataManager(player_data, heat_map_data)
        self.init_batch_managers(match_data, train_perc)
        self.init_tensors()
        self.init_model()
        self.saver = tf.train.Saver()

    def init_tensors(self):
        self.match_matricies = tf.placeholder(tf.float32, [self.batch_size, self.game_dim[0], self.game_dim[1], self.num_player_attributes])
        self.results = tf.placeholder(tf.float32, [self.batch_size, 3])

    def init_model(self):
        self.model = ConvNetwork(self.model_config)
        self.y_pred = self.model.train()
        self.loss = self.model.loss(self.y_pred, self.results)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self):
        eval_freq = 100
        save_freq = 1000
        test_eval_batches = 20
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
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

