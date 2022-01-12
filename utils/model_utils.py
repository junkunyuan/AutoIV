import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def get_tf_var(names):
    """ Get all trainable variables. """

    _vars = []
    for na_i in range(len(names)):
        _vars = _vars + \
            tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=names[na_i])
    return _vars


def get_opt(lrate, NUM_ITER_PER_DECAY, lrate_decay, loss, _vars):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(
        lrate, global_step, NUM_ITER_PER_DECAY, lrate_decay, staircase=True)
    opt = tf.compat.v1.train.AdamOptimizer(lr)
    train_opt = opt.minimize(loss, global_step=global_step, var_list=_vars)
    return train_opt


def get_var(_dim_in, _dim_out, _name, get_flag=False):
    if get_flag:
        var = tf.get_variable(name=_name, shape=[_dim_in, _dim_out],
                              initializer=tf.contrib.layers.xavier_initializer())
    else:
        var = tf.Variable(tf.random.normal(
            [_dim_in, _dim_out], stddev=0.1 / np.sqrt(_dim_out)), name=_name)
    return var


class MnistEmb(object):
    def __init__(self, confounder_dim, name):
        """Train MNIST classifiler."""

        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.c_dim = confounder_dim
        self.name = name
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.build_classifier()

    def weight_var(self, _shape):
        return tf.Variable(tf.truncated_normal(_shape, stddev=0.1))

    def conv2d_basic(self, _x, _w_shape, b_shape, _name):
        Weights = tf.Variable(tf.truncated_normal(
            _w_shape, stddev=0.1), name='W_' + _name)
        biases = tf.Variable(tf.constant(
            0.1, shape=b_shape), name='b_' + _name)
        return tf.nn.relu(tf.nn.conv2d(_x, Weights, strides=[1, 1, 1, 1], padding="SAME") + biases)

    def build_classifier(self):
        """Layer1: Convolution and Pooling."""
        h_conv1 = self.conv2d_basic(
            self.x_image, [5, 5, 1, 32], [32], _name="conv1")
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[
                                 1, 2, 2, 1], padding="SAME")

        """Layer 2: Convolution and Pooling."""
        h_conv2 = self.conv2d_basic(
            h_pool1, [5, 5, 32, 64], [64], _name="conv2")
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[
                                 1, 2, 2, 1], padding="SAME")

        """Layer 3: Fully-connected."""
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        W_fc1, b_fc1 = get_var(7 * 7 * 64, 512, 'W_fc1' +
                               self.name, True), self.weight_var([512])
        h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        """Layer 4: Fully-connected."""
        W_fc2, b_fc2 = get_var(512, 64, 'W_fc2' + self.name,
                               True), self.weight_var([64])
        h_fc2 = tf.tanh(tf.nn.elu(tf.matmul(h_fc1, W_fc2) + b_fc2))

        """Layer 5: Fully-connected => embedding."""
        W_fc3, b_fc3 = get_var(64, self.c_dim, 'W_fc3' +
                               self.name, True), self.weight_var([self.c_dim])
        # self.embedding = tf.tanh(tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3))
        self.embedding = tf.tanh(tf.matmul(h_fc2, W_fc3) + b_fc3)

        """Layer 6: Fully-connected."""
        W_fc4, b_fc4 = get_var(self.c_dim, 10, 'W_fc4' +
                               self.name, True), self.weight_var([10])
        self.pred = tf.nn.softmax(tf.matmul(self.embedding, W_fc4) + b_fc4)

        """Loss, accuracy, and train_opt."""
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y *
                                   tf.log(self.pred), reduction_indices=[1]))
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1)), tf.float32))
        self.train_opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def train(self, epochs=1000, batch_size=1000):
        """Train MNIST classifier."""

        interval = epochs // 10
        print('=' * 50 + '\nStart to train MNIST classifier.')
        self.sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            data = self.mnist.train.next_batch(batch_size)
            if i % interval == 0 or i == epochs - 1:
                acc_train, loss = self.sess.run([self.acc, self.loss], feed_dict={
                                                self.x: data[0], self.y: data[1]})
                print("Step %d, acc_train: %g, loss: %g" %
                      (i, acc_train, loss))
            self.sess.run(self.train_opt, feed_dict={
                          self.x: data[0], self.y: data[1]})
        # print(self.sess.run(self.embedding, feed_dict={self.x: data[0], self.y: data[1]}))
        acc_test = self.sess.run(self.acc, feed_dict={
                                 self.x: self.mnist.test.images, self.y: self.mnist.test.labels})
        print('\nAcc_test: %g' % acc_test)
        print('Finish training MNIST classifier.\n' + '=' * 50)

    def get_emb(self, digits):
        """ Get embedding data after training. """
        if type(digits) is int:
            data = self.mnist.train.next_batch(digits)
            emb_data = self.sess.run(self.embedding, feed_dict={
                                     self.x: data[0], self.y: data[1]})
            return emb_data

        def get_dig(n, digit):
            temp = self.mnist.train.next_batch(500)
            temp_image, temp_label = temp[0], temp[1]
            ind = np.where(np.argmax(temp_label, axis=1) == digit)[0]
            temp_image = temp_image[ind][:n]
            temp_label = temp_label[ind][:n]
            return self.sess.run(self.embedding, feed_dict={self.x: temp_image, self.y: temp_label})

        emb_data = get_dig(1, digits[0][0])
        for i in range(1, digits.shape[0]):
            data_temp = get_dig(1, digits[i][0])
            emb_data = np.concatenate([emb_data, data_temp], axis=0)
        return emb_data
