import numpy as np
import os
import csv
from matplotlib import pyplot as plt

from utils.utils import create_path, get_line
from utils.model_utils import MnistEmb

import os
import numpy as np
import tensorflow as tf
import random


def norma(x):
    mean, std = np.mean(x), np.std(x)
    return (x - mean) / std


def show_data(_x, _name, file=None, print_flag=False, save_flag=True):
    """Print and save data distribution."""
    meanx, stdx, minx, maxx = np.mean(_x), np.std(_x), np.min(_x), np.max(_x)
    if print_flag:
        print('Shape of {}: {}'.format(_name, ', '.join(_x.shape)))
        print('Mean and std of {}: {}+-{}'.format(_name, meanx, stdx))
        print('Min and max of {}: {}, {}\n'.format(_name, minx, maxx))
    if save_flag and (file is not None):
        with open(file, 'a') as f:
            f.write('Shape of {}: {}\n'.format(_name, str(_x.shape)))
            f.write('Mean and std of {}: {}+-{}\n'.format(_name, meanx, stdx))
            f.write('Min and max of {}: {}, {}\n\n'.format(_name, minx, maxx))


def mnist(all_paras, get_data):
    """Set paras."""
    num = all_paras['train'] + all_paras['valid'] + all_paras['test']
    exp_num = all_paras['exp_num']

    """Create folder for data."""
    create_path(all_paras['data_path'])

    # A folder for saving information.
    info_path = all_paras['data_path'] + 'info/'
    if not os.path.exists(info_path):
        os.mkdir(info_path)

    # A folder for saving csv and npz data.
    data_path = all_paras['data_path'] + 'data/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    dim_zrca = {'dim_z': 10, 'dim_u': 1, 'dim_f': 10, 'dim_a': 4}

    """ If MNIST dataset, train MNIST classifiers. """
    if all_paras['dataset'] == 'mnistz':
        mnist_model_z = MnistEmb(confounder_dim=dim_zrca['dim_z'], name='z')
        mnist_model_z.train()
    elif all_paras['dataset'] == 'mnistc':
        mnist_model_c = MnistEmb(confounder_dim=dim_zrca['dim_f'], name='c')
        mnist_model_c.train()
    elif all_paras['dataset'] == 'mnistzc':
        mnist_model_z = MnistEmb(confounder_dim=dim_zrca['dim_z'], name='z')
        mnist_model_z.train()
        mnist_model_c = MnistEmb(confounder_dim=dim_zrca['dim_f'], name='c')
        mnist_model_c.train()

    datas = []
    print('=' * 50 + '\nStart to generate new data.')
    print_set = False
    for exp_i in range(exp_num):

        file_save = info_path + 'x_y_ye-train_{}-distribution_{}.txt'.format(
            all_paras['train'], exp_i + 1)
        with open(file_save, 'w') as f:
            f.write('Data distribution information of exp {}: \n'.format(exp_i + 1))

        """Generate z."""
        if (all_paras['dataset'] == 'mnistz') or (all_paras['dataset'] == 'mnistzc'):
            z = mnist_model_z.get_emb(num)
        else:
            z = np.random.normal(size=(num, dim_zrca['dim_z']))
        show_data(z, 'z', file_save, print_flag=print_set)

        """Generate rest variable."""
        u = np.random.normal(size=(num, dim_zrca['dim_u']))
        show_data(u, 'u', file_save, print_flag=print_set)

        """Generate confounder variable."""
        if (all_paras['dataset'] == 'mnistc') or (all_paras['dataset'] == 'mnistzc'):
            f = mnist_model_c.get_emb(num)
        else:
            f = np.random.normal(size=(num, dim_zrca['dim_f']))
        show_data(f, 'f', file_save, print_flag=print_set)

        """Generate adjustment variable."""
        a = np.random.normal(size=(num, dim_zrca['dim_a']))
        show_data(a, 'a', file_save, print_flag=print_set)

        """Generate e."""
        e = np.random.normal(size=(num, 1))
        show_data(e, 'e', file_save, print_flag=print_set)

        """Generate x."""
        iv_strength = 0.5
        x = 2. * e * (1 - iv_strength)
        fx_coef = 0.5
        for z_i in range(dim_zrca['dim_z']):
            x = x + 2 * iv_strength * z[:, z_i:(z_i + 1)] / dim_zrca['dim_z']
        for f_i in range(dim_zrca['dim_f']):
            x = x + fx_coef * f[:, f_i:(f_i + 1)] / dim_zrca['dim_f']
        show_data(x, 'x', file_save, print_flag=print_set)

        """Generate y."""
        g = np.abs(x)
        y = g
        y = norma(y)

        """Generate ye."""
        ye = g + 2. * e
        fy_coef, ay_coef = 0.4, 0.2
        for f_i in range(dim_zrca['dim_f']):
            ye = ye + fy_coef * f[:, f_i:(f_i + 1)] / dim_zrca['dim_f']
        for a_i in range(dim_zrca['dim_a']):
            ye = ye + ay_coef * a[:, a_i:(a_i + 1)] / dim_zrca['dim_a']
        ye = norma(ye)
        show_data(ye, 'ye', file_save, print_flag=print_set)

        """Write data in csv."""
        with open(data_path + 'exp{}.csv'.format(exp_i), 'w', newline='') as file:
            file.write('x, y, ye\n')
            csv_writer = csv.writer(file, delimiter=',')
            for j in range(num):
                temp = [x[j][0], y[j][0], ye[j][0]]
                csv_writer.writerow(temp)

        c = np.concatenate([u, f, a], axis=1)
        v = np.concatenate([z, c], axis=1)

        """Write data in npz."""

        np.savez(data_path + 'exp{}.npz'.format(exp_i), v=v, z=z, c=c, x=x, y=y, ye=ye, exp_num=exp_num,
                 train=all_paras['train'], valid=all_paras['valid'], test=all_paras['test'])

        if get_data:
            datas = datas + [{'v': v, 'z': z, 'c': c, 'x': x, 'y': y, 'ye': ye, 'exp_num': exp_num,
                              'train': all_paras['train'], 'valid': all_paras['valid'], 'test': all_paras['test']}]

        print('Finish generating {}th data'.format(exp_i + 1))

    print('Data size:\n\tz: {}\n\tc: {}\n\tx: {}\n\ty: {}'.format(
        z.shape, c.shape, x.shape, y.shape))
    print('Finish generating and saving data.\n' + '=' * 50)

    """ Return data or path. """
    if get_data:
        return datas
    else:
        return all_paras['data_path']
