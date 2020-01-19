import tensorflow as tf

class ConvNetwork(tf.keras.Model):
    def __init__(self, model_config):
        super(ConvNetwork, self).__init__()
        self.config = model_config
        self.build_network(model_config)

    def build_network(self, model_config):
        num_conv_layers = len(model_config['conv_layers']['kernals'])
        conv_layers = []
        for i in range(num_conv_layers):
            kernals = model_config['conv_layers']['kernals'][i]
            kernal_size = model_config['conv_layers']['kernal_size'][i]
            conv_layers.append(tf.keras.layers.Conv2D(kernals, kernel_size=kernal_size, activation=tf.nn.leaky_relu))
            conv_layers.append(tf.keras.layers.MaxPool2D())

        flatten = tf.keras.layers.Flatten()

        num_dense_layers = len(model_config['dense_layers'])
        hidden_layers = [tf.keras.layers.Dense(model_config['dense_layers'][i], activation=tf.nn.leaky_relu) for i in range(num_dense_layers)]

        final_layer = tf.keras.layers.Dense(3, activation=tf.nn.softmax)

        self.network = conv_layers + [flatten] + hidden_layers + [final_layer]
    def call(self, input_matrix):
        input = tf.transpose(input_matrix, perm=[0, 2, 3, 1])
        for layer in self.network:
            input = layer(input)
        y_pred = input
        return y_pred