import os
import tensorflow as tf
import numpy as np

from utils.utils import *
from utils.model_utils import *

from generator.data_generator import gen_load_data
from models.AutoIV_model import AutoIV


class TrainSess(object):
    def __init__(self, model, train_opts, train_steps, data, all_paras, log):
        """Train tensorflow model."""

        self.model, self.train_opts, self.train_steps = model, train_opts, train_steps
        self.data = data
        self.all_paras = all_paras
        self.log = log
        self.epochs, self.int = all_paras['epochs'], all_paras['epochs'] // all_paras['interval']
        self.train_mses, self.valid_mses, self.test_mses = [], [], []
        self.detail_file = self.log.get_path('detail')

    def train(self):
        exp_log(sta_end_flag='start', log=self.log, all_paras=self.all_paras)
        for exp in range(self.all_paras['exp_num']):
            self.model.sess.run(tf.compat.v1.global_variables_initializer())
            self.train_mse_int, self.valid_mse_int, self.test_mse_int = [], [], []

            """Load data."""
            data = self.data[exp]
            v, x, y, ye = data['v'], data['x'], data['y'], data['ye']
            data_split = [self.all_paras['train'],
                          self.all_paras['valid'], self.all_paras['test']]
            train_range = range(0, data_split[0])
            valid_range = range(data_split[0], data_split[0] + data_split[1])
            test_range = range(
                data_split[0] + data_split[1], data_split[0] + data_split[1] + data_split[2])

            """Training, validation, and test data."""
            v_train, x_train, ye_train = v[train_range,
                                           :], x[train_range, :], ye[train_range, :]
            v_valid, x_valid, ye_valid = v[valid_range,
                                           :], x[valid_range, :], ye[valid_range, :]
            v_test, x_test, y_test = v[test_range,
                                       :], x[test_range, :], y[test_range, :]

            """Training, validation, and test dict."""
            dict_train_true = {self.model.v: v_train, self.model.x: x_train, self.model.y: ye_train,
                               self.model.train_flag: True}
            dict_train = {self.model.v: v_train, self.model.x: x_train, self.model.x_pre: x_train,
                          self.model.y: ye_train, self.model.train_flag: False}
            dict_valid = {self.model.v: v_valid, self.model.x: x_valid, self.model.x_pre: x_valid,
                          self.model.y: ye_valid, self.model.train_flag: False}
            dict_test = {self.model.v: v_test, self.model.x_pre: x_test, self.model.y: y_test,
                         self.model.train_flag: False}

            """Train model."""
            self.log.write(
                'detail', '=' * 50 + '\nStart {}th experiment.'.format(exp + 1), _print_flag=True)
            for ep_th in range(self.epochs):
                if (ep_th % self.int == 0) or (ep_th == self.epochs - 1):
                    loss = self.model.sess.run([self.model.loss_cx2y,
                                                self.model.loss_zc2x,
                                                self.model.lld_zx,
                                                self.model.lld_zy,
                                                self.model.lld_cx,
                                                self.model.lld_cy,
                                                self.model.lld_zc,
                                                self.model.bound_zx,
                                                self.model.bound_zy,
                                                self.model.bound_cx,
                                                self.model.bound_cy,
                                                self.model.bound_zc,
                                                self.model.loss_reg],
                                               feed_dict=dict_train)

                    self.log.write('detail', 'Epoch {}th:'.format(
                        str(ep_th).zfill(4)), _print_flag=True)
                    coef_name = [key for key in self.all_paras['coefs']]
                    for i in range(len(loss)):
                        self.log.write('detail', '\tLoss_{}: %.6f'.format(
                            coef_name[i][5:]) % loss[i], _print_flag=True)

                    """Get train and valid mse."""
                    y_pre_train = self.model.sess.run(
                        self.model.y_pre, feed_dict=dict_train)
                    y_pre_valid = self.model.sess.run(
                        self.model.y_pre, feed_dict=dict_valid)
                    y_pre_test = self.model.sess.run(
                        self.model.y_pre, feed_dict=dict_test)

                    mse_train = np.mean(np.square(y_pre_train - ye_train))
                    mse_valid = np.mean(np.square(y_pre_valid - ye_valid))
                    mse_test = np.mean(np.square(y_pre_test - y_test))

                    """Save mse."""
                    self.log.write('detail', '-' * 50 + '\n\ttrain: %.4f | valid: %.4f | test: %.4f\n'
                                   % (float(mse_train), float(mse_valid), float(mse_test)), _print_flag=True)

                    self.train_mse_int = np.append(
                        self.train_mse_int, mse_train)
                    self.valid_mse_int = np.append(
                        self.valid_mse_int, mse_valid)
                    self.test_mse_int = np.append(self.test_mse_int, mse_test)

                for i in range(len(self.train_opts)):  # optimizer to train
                    for j in range(self.train_steps[i]):  # steps of optimizer
                        self.model.sess.run(
                            self.train_opts[i], feed_dict=dict_train_true)

            """Save final MSE results."""
            self.train_mses = np.append(self.train_mses, mse_train)
            self.valid_mses = np.append(self.valid_mses, mse_valid)
            self.test_mses = np.append(self.test_mses, mse_test)

            """Save variables after training."""
            z, c = data['z'], data['c']
            v_c0 = np.concatenate(
                [z, np.zeros((c.shape[0], c.shape[1]))], axis=1)
            dict_all = {self.model.v: v, self.model.x: x,
                        self.model.y: y, self.model.train_flag: False}
            dict_all_c0 = {self.model.v: v_c0, self.model.train_flag: False}
            res_val_save(self.model, self.all_paras, [
                         dict_all, dict_all_c0], exp)

        exp_log('end', self.log, self.all_paras)

        return [self.train_mses, self.valid_mses, self.test_mses], loss


def run(all_paras):
    """Run AutoIV."""

    """Create result files."""
    log = Log(all_paras)

    """Get data."""
    data = gen_load_data(all_paras, get_data=True)

    """Get model."""
    tf.compat.v1.reset_default_graph()
    dim_x, dim_v, dim_y = data[0]['x'].shape[1], data[0]['v'].shape[1], data[0]['y'].shape[1]
    model = AutoIV(all_paras, dim_x, dim_v, dim_y)

    """Get trainable variables."""
    zx_vars = get_tf_var(['zx'])
    zy_vars = get_tf_var(['zy'])
    cx_vars = get_tf_var(['cx'])
    cy_vars = get_tf_var(['cy'])
    zc_vars = get_tf_var(['zc'])
    rep_vars = get_tf_var(['rep/rep_z', 'rep/rep_c'])
    x_vars = get_tf_var(['x'])
    emb_vars = get_tf_var(['emb'])
    y_vars = get_tf_var(['y'])

    vars_lld = zx_vars + zy_vars + cx_vars + cy_vars + zc_vars
    vars_bound = rep_vars
    vars_2stage = rep_vars + x_vars + emb_vars + y_vars

    """Set optimizer."""
    train_opt_lld = get_opt(lrate=all_paras['lrate'], NUM_ITER_PER_DECAY=100,
                            lrate_decay=0.95, loss=model.loss_lld, _vars=vars_lld)

    train_opt_bound = get_opt(lrate=all_paras['lrate'], NUM_ITER_PER_DECAY=100,
                              lrate_decay=0.95, loss=model.loss_bound, _vars=vars_bound)

    train_opt_2stage = get_opt(lrate=all_paras['lrate'], NUM_ITER_PER_DECAY=100,
                               lrate_decay=0.95, loss=model.loss_2stage, _vars=vars_2stage)

    train_opts = [train_opt_lld, train_opt_bound, train_opt_2stage]
    train_steps = [all_paras['opt_lld_step'],
                   all_paras['opt_bound_step'], all_paras['opt_2stage_step']]

    ''' Run experiments '''
    train_sess = TrainSess(
        model, train_opts, train_steps, data, all_paras, log)
    result, _ = train_sess.train()

    return result, log
