import tensorflow as tf

class ConvNetwork():
    def __init__(self, model_config):
        self.config = model_config

    def build_network(self, input_matrix):
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation=tf.nn.leaky_relu)(input_matrix)
        pool1 = tf.keras.layers.MaxPool2D()(conv1)

        conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(pool1)
        pool2 = tf.keras.layers.MaxPool2D()(conv2)

        flatten = tf.keras.layers.Flatten(pool2)

        hidden_layer1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(flatten)
        hidden_layer2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)(hidden_layer1)
        hidden_layer3 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(hidden_layer2)
        hidden_layer4 = tf.keras.layers.Dense(3)(hidden_layer3)

        return hidden_layer4


    def call(self, input_matrix, y):
        y_pred = self.build_network(input_matrix)
        loss = tf.nn.softmax_cross_entropy_with_logits(y_pred, y)
        return y_pred, loss