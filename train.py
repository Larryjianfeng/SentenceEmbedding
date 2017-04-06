#encoding=utf-8
import os
import itertools
import numpy as np
import tensorflow as tf
from reader import TextReader
import random
import time

embed_dim = 256
h_dim = 256
k = 100

data_path = './'
model_dir = './model_dir/'
reader = TextReader(data_path)

def create_train_op(loss, var_list):
    train_op = tf.contrib.layers.optimize_loss(loss = loss,
        global_step = tf.contrib.framework.get_global_step(),
        learning_rate = 0.01,
        clip_gradients = 10.0,
        optimizer = "Adam",
        variables = var_list)
    return train_op


global_step = tf.Variable(0, name = 'global_step', trainable=False)

tx = tf.placeholder(tf.int64, [None, reader.vocab_size])
x = tf.to_float(tx)

batch_size = tf.placeholder(tf.int64)

with tf.variable_scope('encoder'):
	w_1 = tf.get_variable('w_1', [reader.vocab_size, embed_dim], initializer = tf.random_uniform_initializer(-1.0, 1.0))
	b_1 = tf.get_variable('b_1', [embed_dim], initializer = tf.truncated_normal_initializer(0, 0.1))

	L1 = tf.nn.bias_add(tf.matmul(x, w_1), b_1)
	L1 = tf.nn.tanh(L1)

	w_2 = tf.get_variable('w_2', [embed_dim, embed_dim], initializer = tf.truncated_normal_initializer(0, 0.1))
	b_2 = tf.get_variable('b_2', [embed_dim], initializer = tf.truncated_normal_initializer(0, 0.1))

	L2 = tf.nn.bias_add(tf.matmul(L1, w_2), b_2)
	L2 = tf.nn.tanh(L2)

	w_encoder_mu = tf.get_variable('w_encoder_mu', [embed_dim, h_dim], initializer = tf.truncated_normal_initializer(0, 0.1))
	b_encoder_mu = tf.get_variable('b_encoder_mu', [h_dim], initializer = tf.truncated_normal_initializer(0, 0.1))


	mu = tf.nn.l2_normalize(tf.nn.bias_add(tf.matmul(L2, w_encoder_mu), b_encoder_mu), 1)


with tf.variable_scope('clustering') as vs_cluster:
    k_means = tf.get_variable('k_means', [h_dim, k], initializer = tf.random_uniform_initializer(-1.0, 1.0))
    k_means = tf.nn.l2_normalize(k_means, 0)
    re = tf.matmul(mu, k_means)
    index_re  = tf.argmax(re, 1)

    cosine_simi = tf.reduce_max(re, 1)
    clust_loss = 1.0 - cosine_simi
    c_loss = tf.reduce_mean(clust_loss)

with tf.variable_scope('decoder') as vs:

    R = tf.get_variable('R', [h_dim, reader.vocab_size], initializer = tf.truncated_normal_initializer(0, 0.01))
    b = tf.get_variable('b', [reader.vocab_size], initializer = tf.truncated_normal_initializer(0, 0.01))

    e = tf.matmul(mu, R) + b
    p_x_i = tf.nn.softmax(e, -1)
    g_loss = -tf.reduce_sum(tf.log(p_x_i + 1e-10)*x, 1)
    g_loss_stand = -tf.log(1.0/tf.reduce_sum(x, 1))*tf.reduce_sum(x, 1)
    loss_2 = tf.reduce_mean(g_loss/g_loss_stand)
    loss = tf.reduce_mean(g_loss)

c_var_list = []
d_var_list = []
for var in tf.trainable_variables():
    if 'clustering' in var.name:
        c_var_list.append(var)
    else:
        d_var_list.append(var)

for i in d_var_list:print i.name

#optim_g = tf.train.AdamOptimizer(learning_rate=0.01).minimize(c_loss, global_step=global_step, var_list=c_var_list)
#optim_all = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, global_step=global_step, var_list=d_var_list)
optim_all = create_train_op(loss, None)
optim_g = create_train_op(c_loss, c_var_list)


saver = tf.train.Saver()
with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print 'the model being restored is '
        print ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        print 'sucesssfully restored the session'

    count = global_step.eval()
    for k in range(0, 100000):
    	data, length = reader.iterator()
        lm, gm, _= sess.run([loss, loss_2, optim_all], feed_dict = {tx: data, batch_size:length})
        print 'After\t' + str(global_step.eval()) + ' th step,the loss\t' + str(gm) + '\t the loss is\t' + str(lm)
        global_step.assign(count).eval()
        if k%100 == 0:
            saver.save(sess, model_dir + 'model.ckpt', global_step = global_step)
        count += 1

    for k in range(0, 0):
        data, length = reader.iterator()
        cm, _ = sess.run([c_loss, optim_g], feed_dict = {tx: data, batch_size:length})
        print 'After\t' + str(global_step.eval()) + ' th step,the c loss\t' + str(cm)
        global_step.assign(count).eval()
        if k%100 == 0:
            saver.save(sess, model_dir + 'model.ckpt', global_step = global_step)
        count += 1

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print 'the model being restored is '
        print ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        print 'sucesssfully restored the session'
    AMU = []
    text = reader.train_data
    data, length = [np.bincount(t, minlength = reader.vocab_size) for t in text] , len(text)
    AMU, topics, simi = sess.run([mu, index_re, cosine_simi] ,feed_dict = {tx: data, batch_size:length})

def nearest(i):
    v0 = AMU[i]
    simi = np.sum(AMU[i]*AMU, -1)
    simi_g = simi.argsort()[::-1]
    for t in range(5):
        print reader.texts[simi_g[t]] + '\t' + str(simi[simi_g[t]])

simi_matrix = np.matmul(AMU, np.transpose(AMU))
epsilon = 0.5


