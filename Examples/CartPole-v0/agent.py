import tensorflow as tf
import tensorflow.contrib.layers as layers
from replay import ReplayBuffer


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


class Agent(object):
    def __init__(self, env):
        self._env = env
        self.sess = tf.Session()
        self.graph = self.build_graph()
        self.replay = ReplayBuffer(50000)
        self.sess.run(tf.global_variables_initializer())

    def def_action(self):
        s = tf.placeholder(tf.float32, [None] + [self._env.observation_space.shape])
        with tf.variable_scope('q_net', reuse=False):
            q = self.network(s)
        act = tf.argmax(q, axis=1)
        return s, act

    def network(self, x):
        y = x
        y = layers.fully_connected(y, num_outputs=64, activation_fn=tf.nn.relu)
        y = layers.fully_connected(y, num_outputs=self._env.action_space.n, activation_fn=None)
        return y

    def build_graph(self):
        act_s, act = self.def_action()

        s = tf.placeholder(tf.float32, [None] + [self._env.observation_space.shape])
        a = tf.placeholder(tf.int32, [None])
        r = tf.placeholder(tf.float32, [None])
        t = tf.placeholder(tf.float32, [None])
        s_ = tf.placeholder(tf.float32, [None] + [self._env.observation_space.shape])

        with tf.variable_scope('q_net', reuse=True):
            q = self.network(s)
        q_var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='q_net')

        with tf.variable_scope('q_target_net', reuse=False):
            q_target = self.network(s_)
        q_target_var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='q_net')

        q_selected = tf.reduce_sum(q * tf.one_hot(a, self._env.action_space.n), 1)
        q_target_best = tf.reduce_max(q_target, 1)

        q_target_best_masked = (1.0 - t) * q_target_best

        q_target_selected = r + 1.0 * q_target_best_masked
        error = q_selected - tf.stop_gradient(q_target_selected)
        loss = huber_loss(error)

        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = minimize_and_clip(opt, loss, q_var)

        update_target_expr = []
        for var, var_target in zip(sorted(q_var, key=lambda v: v.name),
                                   sorted(q_target_var, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        return {'act_s': act_s, 'act': act, 's': s, 'a': a, 'r': r, 't': t, 's_': s_, 'train_op': train_op,
                'update_target_expr': update_target_expr}
