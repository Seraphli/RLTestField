import tensorflow as tf, pickle, numpy as np

sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
ops = graph.get_operations()
with open('batch_1', 'rb') as f:
    batch = pickle.load(f)
obses_t, actions, rewards, obses_tp1, dones = batch
weights = np.ones_like(rewards)
loss = graph.get_tensor_by_name('deepq_1/Mean:0')
s = graph.get_tensor_by_name('deepq_1/obs_t:0')
a = graph.get_tensor_by_name('deepq_1/action:0')
r = graph.get_tensor_by_name('deepq_1/reward:0')
s_ = graph.get_tensor_by_name('deepq_1/obs_tp1:0')
t = graph.get_tensor_by_name('deepq_1/done:0')
w = graph.get_tensor_by_name('deepq_1/weight:0')
loss_value = sess.run(loss, feed_dict={s: obses_t, a: actions, r: rewards, s_: obses_tp1, t: dones, w: weights})
print(loss_value)
