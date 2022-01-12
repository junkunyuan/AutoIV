import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from utils.model_utils import get_var
import os


class AutoIV(object):
    def __init__(self, all_paras, dim_x, dim_v, dim_y):
        """Build AutoIV model."""

        """Get sess and placeholder."""
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.dim_x, self.dim_v, self.dim_y = dim_x, dim_v, dim_y
        self.x = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.dim_x], name='x')
        self.v = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.dim_v], name='v')
        self.y = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.dim_y], name='y')

        """Set up parameters."""
        self.emb_dim = all_paras['emb_dim']
        self.rep_dim = all_paras['rep_dim']
        self.num = all_paras['train']
        self.coefs = all_paras['coefs']
        self.dropout = all_paras['dropout']
        self.train_flag = tf.compat.v1.placeholder(tf.bool, name='train_flag')
        self.all_paras = all_paras
        self.get_flag = True  # tf.Variable or tf.get_variable
        self.init = tf.contrib.layers.xavier_initializer()

        """Build model and get loss."""
        self.build_model()
        self.calculate_loss()

    def build_model(self):
        """Build model."""

        """Build representation network."""
        with tf.compat.v1.variable_scope('rep'):
            rep_net_layer = self.all_paras['rep_net_layer']
            self.rep_z, self.w_z, self.b_z = self.rep_net(inp=self.v,
                                                          dim_in=self.dim_v,
                                                          dim_out=self.rep_dim,
                                                          layer=rep_net_layer,
                                                          name='rep_z')
            self.rep_c, self.w_c, self.b_c = self.rep_net(inp=self.v,
                                                          dim_in=self.dim_v,
                                                          dim_out=self.rep_dim,
                                                          layer=rep_net_layer,
                                                          name='rep_c')
        self.rep_zc = tf.concat([self.rep_z, self.rep_c], 1)

        """Build treatment prediction network."""
        with tf.compat.v1.variable_scope('x'):
            self.x_pre, self.w_x, self.b_x = self.x_net(inp=self.rep_zc,
                                                        dim_in=self.rep_dim * 2,
                                                        dim_out=self.dim_x,
                                                        layer=self.all_paras['x_net_layer'])

        """Build embedding network."""
        with tf.compat.v1.variable_scope('emb'):
            self.x_emb, self.w_emb, self.b_emb = self.emb_net(inp=self.x_pre,
                                                              dim_in=self.dim_x,
                                                              dim_out=self.emb_dim,
                                                              layer=self.all_paras['emb_net_layer'])
        self.rep_cx = tf.concat([self.rep_c, self.x_emb], 1)

        """Build outcome prediction network."""
        with tf.compat.v1.variable_scope('y'):
            self.y_pre, self.w_y, self.b_y = self.y_net(inp=self.rep_cx,
                                                        dim_in=self.rep_dim + self.emb_dim,
                                                        dim_out=self.dim_y,
                                                        layer=self.all_paras['y_net_layer'])

        """Maximize MI between z and x."""
        with tf.compat.v1.variable_scope('zx'):
            self.lld_zx, self.bound_zx, self.mu_zx, self.logvar_zx, self.ws_zx = self.mi_net(
                inp=self.rep_z,
                outp=self.x,
                dim_in=self.rep_dim,
                dim_out=self.dim_x,
                mi_min_max='max')

        """Minimize MI between z and y given x."""
        with tf.compat.v1.variable_scope('zy'):
            self.lld_zy, self.bound_zy, self.mu_zy, self.logvar_zy, self.ws_zy = self.mi_net(
                inp=self.rep_z,
                outp=self.y,
                dim_in=self.rep_dim,
                dim_out=self.dim_y,
                mi_min_max='min',
                name='zy')

        """Maximize MI between c and x."""
        with tf.compat.v1.variable_scope('cx'):
            self.lld_cx, self.bound_cx, self.mu_cx, self.logvar_cx, self.ws_cx = self.mi_net(
                self.rep_c,
                outp=self.x,
                dim_in=self.rep_dim,
                dim_out=self.dim_x,
                mi_min_max='max')

        """Maximize MI between c and y."""
        with tf.compat.v1.variable_scope('cy'):
            self.lld_cy, self.bound_cy, self.mu_cy, self.logvar_cy, self.ws_cy = self.mi_net(
                inp=self.rep_c,
                outp=self.y,
                dim_in=self.rep_dim,
                dim_out=self.dim_y,
                mi_min_max='max')

        """Minimize MI between z and c."""
        with tf.compat.v1.variable_scope('zc'):
            self.lld_zc, self.bound_zc, self.mu_zc, self.logvar_zc, self.ws_zc = self.mi_net(
                inp=self.rep_z,
                outp=self.rep_c,
                dim_in=self.rep_dim,
                dim_out=self.rep_dim,
                mi_min_max='min')

    def calculate_loss(self):
        """Get loss."""

        """Loss of y prediction."""
        self.loss_cx2y = tf.reduce_mean(tf.square(self.y - self.y_pre))

        """Loss of t prediction."""
        self.loss_zc2x = tf.reduce_mean(tf.square(self.x - self.x_pre))

        """Loss of network regularization."""
        def w_reg(w):
            """Calculate l2 loss of network weight."""
            w_reg_sum = 0
            for w_i in range(len(w)):
                w_reg_sum = w_reg_sum + tf.nn.l2_loss(w[w_i])
            return w_reg_sum
        self.loss_reg = (w_reg(self.w_z) + w_reg(self.w_c) +
                         w_reg(self.w_emb) + w_reg(self.w_x) + w_reg(self.w_y)) / 5.

        """Losses."""
        self.loss_lld = self.coefs['coef_lld_zy'] * self.lld_zy + \
            self.coefs['coef_lld_cx'] * self.lld_cx + \
            self.coefs['coef_lld_zx'] * self.lld_zx + \
            self.coefs['coef_lld_cy'] * self.lld_cy + \
            self.coefs['coef_lld_zc'] * self.lld_zc

        self.loss_bound = self.coefs['coef_bound_zy'] * self.bound_zy + \
            self.coefs['coef_bound_cx'] * self.bound_cx + \
            self.coefs['coef_bound_zx'] * self.bound_zx + \
            self.coefs['coef_bound_cy'] * self.bound_cy + \
            self.coefs['coef_bound_zc'] * self.bound_zc + \
            self.coefs['coef_reg'] * self.loss_reg

        self.loss_2stage = self.coefs['coef_cx2y'] * self.loss_cx2y + \
            self.coefs['coef_zc2x'] * self.loss_zc2x + \
            self.coefs['coef_reg'] * self.loss_reg

    def layer_out(self, inp, w, b, flag):
        """Set up activation function and dropout for layers."""
        out = tf.matmul(inp, w) + b
        if flag:
            return tf.layers.dropout(tf.nn.elu(out), rate=self.dropout, training=self.train_flag)
        else:
            return out

    def rep_net(self, inp, dim_in, dim_out, layer, name):
        """Representation network."""
        rep, w_, b_ = [inp], [], []
        with tf.compat.v1.variable_scope(name):
            for i in range(layer):
                dim_in_net = dim_in if (i == 0) else dim_out
                dim_out_net = dim_out
                w_.append(get_var(dim_in_net, dim_out_net, 'w_' +
                          name + '_%d' % i, get_flag=self.get_flag))
                b_.append(tf.Variable(
                    tf.zeros([1, dim_out_net]), name='b_' + name + '_%d' % i))
                rep.append(self.layer_out(
                    rep[i], w_[i], b_[i], flag=(i != layer - 1)))
        return rep[-1], w_, b_

    def x_net(self, inp, dim_in, dim_out, layer):
        """Treatment prediction network."""
        x_pre, w_x, b_x = [inp], [], []
        for i in range(layer):
            dim_in_net = dim_in if (i == 0) else dim_in // (i * 2)
            dim_out_net = dim_in // ((i + 1) *
                                     2) if i != (layer - 1) else dim_out
            dim_in_net = dim_in_net if dim_in_net > 0 else 1
            dim_out_net = dim_out_net if dim_out_net > 0 else 1
            w_x.append(get_var(dim_in_net, dim_out_net, 'w_x' +
                       '_%d' % i, get_flag=self.get_flag))
            b_x.append(tf.Variable(
                tf.zeros([1, dim_out_net]), name='b_x' + '_%d' % i))
            x_pre.append(self.layer_out(
                x_pre[i], w_x[i], b_x[i], flag=(i != layer - 1)))
        return x_pre[-1], w_x, b_x

    def emb_net(self, inp, dim_in, dim_out, layer):
        """Treatment embedding network."""
        x_emb, w_emb, b_emb = [inp], [], []
        for i in range(layer):
            dim_in_net = dim_in if (i == 0) else dim_out
            dim_out_net = dim_out
            w_emb.append(get_var(dim_in_net, dim_out_net,
                         'w_emb_%d' % i, get_flag=self.get_flag))
            b_emb.append(tf.Variable(
                tf.zeros([1, dim_out_net]), name='b_emb_%d' % i))
            x_emb.append(self.layer_out(
                x_emb[i], w_emb[i], b_emb[i], flag=(i != layer - 1)))
        return x_emb[-1], w_emb, b_emb

    def y_net(self, inp, dim_in, dim_out, layer):
        """Outcome prediction network."""
        y_pre, w_y, b_y = [inp], [], []
        for i in range(layer):
            dim_in_net = dim_in if (i == 0) else dim_in // (i * 2)
            dim_out_net = dim_in // ((i + 1) *
                                     2) if i != (layer - 1) else dim_out
            dim_in_net = dim_in_net if dim_in_net > 0 else 1
            dim_out_net = dim_out_net if dim_out_net > 0 else 1
            w_y.append(get_var(dim_in_net, dim_out_net, 'w_y' +
                       '_%d' % i, get_flag=self.get_flag))
            b_y.append(tf.Variable(
                tf.zeros([1, dim_out_net]), name='b_y' + '_%d' % i))
            y_pre.append(self.layer_out(
                y_pre[i], w_y[i], b_y[i], flag=(i != layer - 1)))
        return y_pre[-1], w_y, b_y

    def fc_net(self, inp, dim_out, act_fun, init):
        """Fully-connected network."""
        return layers.fully_connected(inputs=inp,
                                      num_outputs=dim_out,
                                      activation_fn=act_fun,
                                      weights_initializer=init)

    def mi_net(self, inp, outp, dim_in, dim_out, mi_min_max, name=None):
        """Mutual information network."""
        h_mu = self.fc_net(inp, dim_in // 2, tf.nn.elu, self.init)
        mu = self.fc_net(h_mu, dim_out, None, self.init)
        h_var = self.fc_net(inp, dim_in // 2, tf.nn.elu, self.init)
        logvar = self.fc_net(h_var, dim_out, tf.nn.tanh, self.init)

        new_order = tf.random_shuffle(tf.range(self.num))
        outp_rand = tf.gather(outp, new_order)

        """Get likelihood."""
        loglikeli = - \
            tf.reduce_mean(tf.reduce_sum(-(outp - mu) ** 2 /
                           tf.exp(logvar) - logvar, axis=-1))

        """Get positive and negative U."""
        pos = - (mu - outp) ** 2 / tf.exp(logvar)
        neg = - (mu - outp_rand) ** 2 / tf.exp(logvar)

        if name == 'zy':
            x_rand = tf.gather(self.x, new_order)

            # Using RBF kernel to measure distance.
            sigma = self.all_paras['sigma']
            w = tf.exp(-tf.square(self.x - x_rand) / (2 * sigma ** 2))
            w_soft = tf.nn.softmax(w, axis=0)
        else:
            w_soft = 1. / self.num

        """Get estimation of mutual information."""
        if mi_min_max == 'min':
            pn = 1.
        elif mi_min_max == 'max':
            pn = -1.
        else:
            raise ValueError
        bound = pn * tf.reduce_sum(w_soft * (pos - neg))

        return loglikeli, bound, mu, logvar, w_soft
