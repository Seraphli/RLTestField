import tensorflow as tf
import tensorflow.contrib.layers as layers


class Agent(object):
    def __init__(self, env):
        self._env = env

    def network(self):
        x = tf.placeholder(tf.float32, [None] + [self._env.observation_space.shape])
        y= x
        y = layers.fully_connected(y, num_outputs=64, activation_fn=tf.nn.relu)
        y = layers.fully_connected(y, num_outputs=self._env.action_space.n, activation_fn=None)
