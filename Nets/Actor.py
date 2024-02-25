import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.models import Model
from keras.optimizers import Adam
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')


class GaussianPolicy(Model):
    def __init__(self, n_actions, layer_size=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.n_actions = n_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layer1 = Dense(layer_size, activation='relu', name="dense1")
        self.layer2 = Dense(layer_size, activation='relu', name="dense2")
        self.layer3 = Dense(layer_size, activation='relu', name="dense3")

        self.mean = Dense(n_actions, activation='linear', name="mean")
        self.log_std = Dense(n_actions, activation='linear', name="std")

        self.conv2d_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu')
        self.conv2d_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu')
        self.flatten = Flatten()
        self.max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))


    def call(self, inputs):

        if len(inputs.shape) == 4:
            inputs = tf.cast(inputs, tf.float64)
            out = self.conv2d_1(inputs)
            out = self.max_pooling(out)
            out = self.conv2d_2(out)
            out = self.max_pooling(out)
            out = self.flatten(out)
            inputs = out

        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)

        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = tf.keras.backend.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample_action(self, state, epsilon=1e-6):
        mean, log_std = self.call(state) # Forward le state => mean, log_std
        std = tf.math.exp(log_std)

        normal = tfp.distributions.MultivariateNormalDiag(mean, std) # distribution action
        z = normal.sample() ## tirage dans loi normale
        action = tf.math.tanh(z) # tanh pour avoir entre -1 et 1
        log_prob_z = normal.log_prob(z)

        if(len(log_prob_z.shape) == 1):
            log_prob_z = tf.expand_dims(log_prob_z, 1)
        log_pi = log_prob_z - tf.reduce_sum(tf.math.log(1 - tf.math.square(action) + epsilon),
                                                          axis=1, keepdims=True)

       
        return action, log_pi
