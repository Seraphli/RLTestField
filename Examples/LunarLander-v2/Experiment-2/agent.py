import tensorflow as tf, os, numpy as np


class Agent(object):
    def __init__(self, state_size, action_size):
        self.state_size, self.action_size = state_size, action_size
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with tf.device('/gpu'):
            self.x, self.y, self.q_target, self.loss, self.train_op = self.network()
        self.sess.run(tf.global_variables_initializer())
        self.step = 0

    def network(self):
        w_init, b_init = tf.contrib.layers.xavier_initializer(seed=12345), tf.constant_initializer(0.01)
        x = tf.placeholder(tf.float32, [None, self.state_size], name='input')
        with tf.variable_scope('hidden1'):
            w = tf.get_variable("w", [self.state_size, 64], initializer=w_init, trainable=True)
            b = tf.get_variable("b", [64], initializer=b_init, trainable=True)
            y = tf.nn.relu(tf.matmul(x, w) + b)
        with tf.variable_scope('hidden2'):
            w = tf.get_variable("w", [64, 32], initializer=w_init, trainable=True)
            b = tf.get_variable("b", [32], initializer=b_init, trainable=True)
            y = tf.nn.relu(tf.matmul(y, w) + b)
        with tf.variable_scope('output'):
            w = tf.get_variable("w", [32, self.action_size], initializer=w_init, trainable=True)
            b = tf.get_variable("b", [self.action_size], initializer=b_init, trainable=True)
            y = tf.matmul(y, w) + b
        q_target = tf.placeholder(tf.float32, [None, self.action_size], name='q_target')
        loss = tf.reduce_mean(tf.squared_difference(q_target, y), name='MSE')
        train_op = tf.train.GradientDescentOptimizer(0.00025).minimize(loss)
        return x, y, q_target, loss, train_op

    def take_action(self, obs, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        q = self.sess.run(self.y, feed_dict={self.x: [obs]})
        return np.argmax(q)

    def eval(self, obs):
        return self.take_action(obs, 0.001)

    def train(self, batch):
        batch_s = []
        batch_q_target = []
        for i in range(len(batch)):
            s, a, r, t, s_ = batch[i]
            batch_s.append(s)
            if t:
                q_target = r
            else:
                q_target = r + 0.99 * np.max(self.sess.run(self.y, feed_dict={self.x: [s_]}))
            q_targets = self.sess.run(self.y, feed_dict={self.x: [s]})[0]
            q_targets[a] = q_target
            batch_q_target.append(q_targets)
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.q_target: batch_q_target, self.x: batch_s})
        self.step += 1
        return loss
