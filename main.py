import os
import numpy as np
import tensorflow as tf
import random
from train import run

import warnings
warnings.filterwarnings('ignore')

FLAGS = tf.compat.v1.app.flags.FLAGS

"""Set up dataset."""
tf.compat.v1.app.flags.DEFINE_string(
    'dataset', 'sin', 'dataset/scenario: sin, step, abs, linear, poly2d, poly3d, mnistz, mnistc, mnistzc')
tf.compat.v1.app.flags.DEFINE_integer('num_train', 500, 'training data')
tf.compat.v1.app.flags.DEFINE_integer('num_valid', 500, 'validation data')
tf.compat.v1.app.flags.DEFINE_integer('num_test', 500, 'test data')

"""Set up training."""
tf.compat.v1.app.flags.DEFINE_integer(
    'rep_net_layer', 2, 'layers of representation networks')
tf.compat.v1.app.flags.DEFINE_integer(
    'x_net_layer', 2, 'layers of treatment prediction network')
tf.compat.v1.app.flags.DEFINE_integer(
    'emb_net_layer', 2, 'layers of embedding network')
tf.compat.v1.app.flags.DEFINE_integer(
    'y_net_layer', 2, 'layers of outcome prediction network')
tf.compat.v1.app.flags.DEFINE_integer('emb_dim', 4, 'embedding dimension')
tf.compat.v1.app.flags.DEFINE_integer('rep_dim', 4, 'representation dimension')
tf.compat.v1.app.flags.DEFINE_float(
    'lrate', 1e-3, 'learning rate of optimizer')
tf.compat.v1.app.flags.DEFINE_float('dropout', 0., 'drop rate of dropout')
tf.compat.v1.app.flags.DEFINE_integer(
    'epochs', 1000, 'training epochs of AutoIV')
tf.compat.v1.app.flags.DEFINE_integer(
    'opt_lld_step', 1, 'steps of likelihood optimizer')
tf.compat.v1.app.flags.DEFINE_integer(
    'opt_bound_step', 1, 'steps of bound optimizer')
tf.compat.v1.app.flags.DEFINE_integer(
    'opt_2stage_step', 1, 'steps of two-stage optimizer')
tf.compat.v1.app.flags.DEFINE_float(
    'sigma', 0.1, 'hyper-parameter sigma of RBF kernel')
tf.compat.v1.app.flags.DEFINE_integer(
    'interval', 2, 'print times during training')
tf.compat.v1.app.flags.DEFINE_integer('exp_num', 3, 'experiment runs')

""" Set up experiments. """
tf.compat.v1.app.flags.DEFINE_boolean(
    'gen_new', False, 'whether generate new data')
tf.compat.v1.app.flags.DEFINE_boolean(
    'del_res', False, 'whether delete all the previous results')
tf.compat.v1.app.flags.DEFINE_string(
    'res_path', 'AutoIV-results/', 'result path')
tf.compat.v1.app.flags.DEFINE_string(
    'res_file', 'summary.csv', 'result summary csv file')
tf.compat.v1.app.flags.DEFINE_string('gpu', '0', 'which GPU to use')
tf.compat.v1.app.flags.DEFINE_integer('seed', 0, 'seed')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

seed = FLAGS.seed
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def main_run(FLAGS):
    """Set hyper-parameters."""
    coefs = {'coef_cx2y': 1, 'coef_zc2x': 1, 'coef_lld_zx': 1,
             'coef_lld_zy': 1, 'coef_lld_cx': 1,  'coef_lld_cy': 1,
             'coef_lld_zc': 1, 'coef_bound_zx': 1, 'coef_bound_zy': 1,
             'coef_bound_cx': 1, 'coef_bound_cy': 1, 'coef_bound_zc': 1, 'coef_reg': 0.001}

    num = {'train': FLAGS.num_train,
           'valid': FLAGS.num_valid, 'test': FLAGS.num_test}

    layers = {'rep_net_layer': FLAGS.rep_net_layer, 'x_net_layer': FLAGS.x_net_layer,
              'emb_net_layer': FLAGS.emb_net_layer, 'y_net_layer': FLAGS.y_net_layer}

    opt_steps = {'opt_lld_step': FLAGS.opt_lld_step, 'opt_bound_step': FLAGS.opt_bound_step,
                 'opt_2stage_step': FLAGS.opt_2stage_step}

    data_path = 'data/{}/{}-train_{}/'.format(
        FLAGS.dataset, FLAGS.dataset, FLAGS.num_train)

    all_paras = {'dataset': FLAGS.dataset, 'data_path': data_path, 'coefs': coefs,
                 'rep_dim': FLAGS.rep_dim, 'lrate': FLAGS.lrate, 'dropout': FLAGS.dropout,
                 'emb_dim': FLAGS.emb_dim, 'epochs': FLAGS.epochs, 'interval': FLAGS.interval,
                 'exp_num': FLAGS.exp_num, 'gen_new': FLAGS.gen_new, 'del_res': FLAGS.del_res,
                 'sigma': FLAGS.sigma, 'res_path': FLAGS.res_path, 'visible_gpu': FLAGS.gpu}

    all_paras.update(num)
    all_paras.update(layers)
    all_paras.update(opt_steps)

    print('\n\n' + '=' * 50)
    print('Run experiment.\n')
    _, _ = run(all_paras)


if __name__ == '__main__':
    main_run(FLAGS)
