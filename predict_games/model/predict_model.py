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
    def __init__(self, match_data_file_name="match_data_epl_18.json", config_name="config.json"):
        np.random.seed(88)
        random.seed(8)


        config = json.load(open(os.path.join(pg.CONFIG_DIR, config_name)))
        self.batch_size = config["batch_size"]
        self.game_dim = config["game_dim"]
        self.player_attributes = config["player_attributes"]
        self.num_player_attributes = len(self.player_attributes)
        self.model_config = config["model_config"]

        self.epochs = config["epochs"]

        match_data = json.load(open(os.path.join(pg.MATCH_DATA_DIR, match_data_file_name)))
        self.data_manager = DataManager(player_attribute_features=self.player_attributes)
        self.init_batch_managers(match_data)
        self.init_model()

    def init_model(self):
        self.model = ConvNetwork(self.model_config)
        self.opt = tf.keras.optimizers.Adam()

    def train(self):
        eval_freq = 100
        test_eval_batches = 20
        train_loss_metric = tf.keras.metrics.Mean()
        train_accuracy_metric = tf.keras.metrics.Accuracy()
        test_loss_metric = tf.keras.metrics.Mean()
        test_accuracy_metric = tf.keras.metrics.Accuracy()
        for i in range(self.epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(self.batch_manager.train_batch_dataset):
                with tf.GradientTape() as tape:
                    y_pred = self.model(x_batch_train)
                    loss = tf.nn.softmax_cross_entropy_with_logits(y_batch_train, y_pred)

                grads = tape.gradient(loss, self.model.trainable_weights)
                self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
                train_loss_metric.update_state(loss)
                train_accuracy_metric.update_state(tf.argmax(y_pred, axis=-1), tf.argmax(y_batch_train, axis=-1))

                if step % eval_freq == 0:
                    for (x_batch_test, y_batch_test) in self.batch_manager.test_batch_dataset.take(test_eval_batches):
                        y_pred = self.model(x_batch_test)
                        loss = tf.nn.softmax_cross_entropy_with_logits(y_batch_test, y_pred)
                        test_loss_metric.update_state(loss)
                        test_accuracy_metric.update_state(tf.argmax(y_pred, axis=-1), tf.argmax(y_batch_test, axis=-1))

                    print('Train ---- Step %s: mean loss = %s accuracy = %s' % (
                    step, train_loss_metric.result().numpy(), train_accuracy_metric.result().numpy()* 100))
                    print('Test ---- Step %s: mean loss = %s accuracy = %s' % (
                    step, test_loss_metric.result().numpy(), test_accuracy_metric.result().numpy()* 100))



    def init_batch_managers(self, match_data):
        self.batch_manager = BatchManager(self.batch_size)
        self.batch_manager.make_batches(match_data, self.data_manager)
