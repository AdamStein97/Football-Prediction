import tensorflow as tf

class ConvNetwork(tf.keras.Model):
    def __init__(self, model_config):
        super(ConvNetwork, self).__init__()
        self.config = model_config
        self.build_network(model_config)

    def build_network(self, model_config):
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation=tf.nn.leaky_relu)
        pool1 = tf.keras.layers.MaxPool2D()

        conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.leaky_relu)
        pool2 = tf.keras.layers.MaxPool2D()

        flatten = tf.keras.layers.Flatten()

        hidden_layer1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        hidden_layer2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
        hidden_layer3 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        hidden_layer4 = tf.keras.layers.Dense(3)

        self.network = [conv1, pool1, conv2, pool2, flatten, hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4]

    def call(self, input_matrix):
        input = tf.transpose(input_matrix, perm=[0, 2, 3, 1])
        for layer in self.network:
            input = layer(input)
        y_pred = input
        return y_pred