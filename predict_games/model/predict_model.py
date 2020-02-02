import os
import predict_games as pg
import json
import pandas as pd
import math
import random
import tensorflow as tf
import numpy as np
import datetime

from predict_games.data_manager import DataManager
from predict_games.batch_manager import BatchManager
from predict_games.model.conv_network import ConvNetwork

class GamePredictModel():
    def __init__(self, match_data_file_names=["match_data_epl_18.json","match_data_epl_19.json","match_data_la_liga_18.json","match_data_la_liga_19.json"], config_name="config.json"):
        np.random.seed(8)
        random.seed(8)
        tf.random.set_seed(8)


        config = json.load(open(os.path.join(pg.CONFIG_DIR, config_name)))
        self.batch_size = config["batch_size"]
        self.player_attributes = config["player_attributes"]
        self.num_player_attributes = len(self.player_attributes)
        self.model_config = config["model_config"]

        self.epochs = config["epochs"]

        match_data = []
        for season in match_data_file_names:
            match_data += json.load(open(os.path.join(pg.MATCH_DATA_DIR, season)))

        self.data_manager = DataManager(player_attribute_features=self.player_attributes)
        self.init_batch_managers(match_data, match_val=json.load(open(os.path.join(pg.MATCH_DATA_DIR, "match_data_epl_20.json"))), data_names=match_data_file_names,val_data_names=["match_data_epl_20.json"])
        self.init_model()

    def init_model(self):
        self.model = ConvNetwork(self.model_config)
        self.opt = tf.keras.optimizers.Adam()


    def train(self):
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=["accuracy"])
        test_loss_metric = tf.keras.metrics.Mean()
        test_accuracy_metric = tf.keras.metrics.Accuracy()

        log_dir = pg.LOG_DIR + "/" + datetime.datetime.now().strftime("%Y%m%d")
        print(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


        self.model.fit(self.batch_manager.train_batch_dataset,
                       epochs=int(self.epochs),
                       validation_data=self.batch_manager.test_batch_dataset,
                       verbose=1,
                       callbacks=[tensorboard_callback]
                       )

        for (x_batch_test, y_batch_test) in self.val_batch_manager.train_batch_dataset:
            y_pred = self.model(x_batch_test)
            loss = tf.keras.losses.categorical_crossentropy(y_batch_test, y_pred)
            test_loss_metric.update_state(loss)
            test_accuracy_metric.update_state(tf.argmax(y_pred, axis=-1), tf.argmax(y_batch_test, axis=-1))

        print("-- Training Completed --")
        print('Test set: mean loss = %s accuracy = %s' % (
            test_loss_metric.result().numpy(), test_accuracy_metric.result().numpy() * 100))


    def train_debug(self):
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
                    loss = tf.keras.losses.categorical_crossentropy(y_batch_train, y_pred)

                grads = tape.gradient(loss, self.model.trainable_weights)
                self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
                train_loss_metric.update_state(loss)
                train_accuracy_metric.update_state(tf.argmax(y_pred, axis=-1), tf.argmax(y_batch_train, axis=-1))

                if step % eval_freq == 0:
                    for (x_batch_test, y_batch_test) in self.batch_manager.test_batch_dataset.take(test_eval_batches):
                        y_pred = self.model(x_batch_test)
                        loss = tf.keras.losses.categorical_crossentropy(y_batch_test, y_pred)
                        test_loss_metric.update_state(loss)
                        test_accuracy_metric.update_state(tf.argmax(y_pred, axis=-1), tf.argmax(y_batch_test, axis=-1))

                    print('Train ---- Step %s: mean loss = %s accuracy = %s' % (
                    step, train_loss_metric.result().numpy(), train_accuracy_metric.result().numpy()* 100))
                    print('Test ---- Step %s: mean loss = %s accuracy = %s' % (
                    step, test_loss_metric.result().numpy(), test_accuracy_metric.result().numpy()* 100))



    def init_batch_managers(self, match_data, data_names,val_data_names, match_val=None):
        if match_val is not None:
            self.val_batch_manager = BatchManager(self.batch_size)
            self.val_batch_manager.make_batches(match_val, self.data_manager, test_perc=0.1, load_data=False,data_names=val_data_names)

        self.batch_manager = BatchManager(self.batch_size)
        self.batch_manager.make_batches(match_data, self.data_manager, load_data=False, data_names=data_names)
