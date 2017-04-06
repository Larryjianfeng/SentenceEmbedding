#encoding=utf-8
import os
import itertools
import numpy as np
import tensorflow as tf
from reader import TextReader
import random 

embed_dim = 512
h_dim = 256


data_path = '/home/yanjianfeng/VAE/n_gram/'
model_dir = '/home/yanjianfeng/VAE/n_gram/model_dir/'
reader = TextReader(data_path)
def create_train_op(loss):
    train_op = tf.contrib.layers.optimize_loss(loss = loss, 
        global_step = tf.contrib.framework.get_global_step(), 
        learning_rate = 0.01, 
        clip_gradients = 10.0, 
        optimizer = "Adam")
    return train_op


global_step = tf.Variable(0, name = 'global_step', trainable=False)

tx = tf.placeholder(tf.int64, [None, reader.vocab_size])
x = tf.to_float(tx)

batch_size = tf.placeholder(tf.int64)
w = tf.placeholder(tf.float32)

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

	w_encoder_var = tf.get_variable('w_encoder_var', [embed_dim, h_dim], initializer = tf.truncated_normal_initializer(0, 0.1))
	b_encoder_var = tf.get_variable('b_encoder_var', [h_dim], initializer = tf.truncated_normal_initializer(0, 0.1))

	mu = tf.nn.bias_add(tf.matmul(L2, w_encoder_mu), b_encoder_mu)
	log_sigma_sq = tf.nn.bias_add(tf.matmul(L2, w_encoder_var), b_encoder_var)

	eps = tf.random_normal([batch_size, h_dim], 0, 1, dtype = tf.float32)
	sigma = tf.sqrt(tf.exp(log_sigma_sq))

	h = mu + sigma*eps

with tf.variable_scope('decoder') as vs:
	R = tf.get_variable('R', [h_dim, reader.vocab_size], initializer = tf.truncated_normal_initializer(0, 0.0001))
	b = tf.get_variable('b', [reader.vocab_size], initializer = tf.truncated_normal_initializer(0, 0.0001))

	e = -tf.matmul(h, R) + b 
	p_x_i = tf.nn.softmax(e, -1)

e_loss = -0.5 * tf.reduce_mean(1.0 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq), 1)
g_loss = -tf.reduce_sum(tf.log(p_x_i + 1e-10)*x, 1)
g_loss_stand = -tf.log(1.0/tf.reduce_sum(x, 1))*tf.reduce_sum(x, 1)
g_loss = g_loss/tf.maximum(g_loss_stand, 1.0)


e_loss_mean = tf.reduce_mean(e_loss)
g_loss_mean = tf.reduce_mean(g_loss)

loss = 0.1*e_loss + g_loss 
loss = tf.reduce_mean(loss)

encoder_var_list = []
decoder_var_list = []
for var in tf.trainable_variables():
    if 'encoder' in var.name:
        encoder_var_list.append(var)
    elif 'decoder' in var.name:
        decoder_var_list.append(var)


optim_e = tf.train.AdamOptimizer(learning_rate=0.05).minimize(e_loss, global_step=global_step, var_list=encoder_var_list)
optim_g = tf.train.AdamOptimizer(learning_rate=0.05).minimize(g_loss, global_step=global_step, var_list=decoder_var_list)
train_op = create_train_op(loss)

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
    for k in range(0, 0):
    	data, length = reader.iterator()
        em, gm, lm, _= sess.run([e_loss_mean, g_loss_mean, loss, train_op], feed_dict = {tx: data, 
            batch_size:length,
            w:k/1000.0})
        print 'After\t' + str(global_step.eval()) + ' th step,the loss\t' + str(lm) + '\t kL loss\t' + str(em) + '\tdecoder loss\t' + str(gm)
        global_step.assign(count).eval()
        if k%1000 == 0:
            saver.save(sess, model_dir + 'model.ckpt', global_step = global_step)
        count += 1

    AMU, AMV = [], []
    for k in range(1):
        text = reader.train_data
        data, length = [np.bincount(t, minlength = reader.vocab_size) for t in text] , 10000
        AM, AU = sess.run([mu, sigma] ,feed_dict = {tx: data, batch_size:length})
        AMU.append(AM)
        AMV.append(AU)
    AMU = [i for j in AMU for i in j]
    AMV = [i for j in AMV for i in j]

def kl_divergence(u0, v0, u1, v1):
    v0, v1 = v0**2, v1**2
    k = len(v0)
    re = v0/v1 + (u1-u0)*(u1-u0)/v1 + np.log(v1/v0)
    return 0.5*(sum(re) - k) 


simi = []
u0, v0 = AMU[2], AMV[2]
for i in range(len(reader.train_data)):
    u1, v1 = AMU[i], AMV[i]
    simi.append(kl_divergence(u0, v0, u1, v1))
simi_s = np.array(simi).argsort()
for i in range(5):
    print simi[simi_s[i]]
    print reader.texts[simi_s[i]]

