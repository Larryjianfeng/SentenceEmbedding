#encoding=utf-8
import os, sys
import numpy as np
import tensorflow as tf
import pickle
import re
os.environ["CUDA_VISIBLE_DEVICES"]=""


embed_dim = 256
h_dim = 256
k = 100

data_path = './n_gram/'
model_dir = './model_dir/'
vocab = pickle.load(open('vocab', 'r'))
vocab_size = len(vocab)

global_step = tf.Variable(0, name = 'global_step', trainable=False)

tx = tf.placeholder(tf.int64, [None, vocab_size])
x = tf.to_float(tx)

batch_size = tf.placeholder(tf.int64)

with tf.variable_scope('encoder'):
	w_1 = tf.get_variable('w_1', [vocab_size, embed_dim], initializer = tf.random_uniform_initializer(-1.0, 1.0))
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




data = open('./train.txt', 'r').read().split('\n')
text = []
for line in data:
    line = line.split(' ')
    line = [l for l in line if re.findall(r'[0-9]', l) == []]
    wd_id = []
    for wd in line:
        try:
            wd_id.append(vocab[wd])
        except:
            continue

    text.append(wd_id)

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
    d, length = [np.bincount(t, minlength = vocab_size) for t in text] , len(text)
    AMU = sess.run([mu] ,feed_dict = {tx: d, batch_size:length})[0]




def query(ques):
    ques = [i.encode('utf-8') for i in list(ques.decode('utf-8'))]
    ques = [i for i in map(vocab.get, ques) if i != None]
    print ques
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        d, length = [np.bincount(t, minlength = vocab_size) for t in [ques]] , 1
        x = sess.run([mu] ,feed_dict = {tx: d, batch_size:length})[0][0]

    simi = np.sum(x*AMU, axis=1)
    r = simi.argsort()[::-1]
    re = []
    i = 0
    re= [data[r[i]], simi[r[i]], r[i]]
    print re[0] + str(re[1])
    print 'answer' + data[re[2]+1]
    return re


re = query('要吃巧克力吗')
