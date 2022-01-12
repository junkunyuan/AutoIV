import os
import numpy as np

from generator.toy_data import toy
from generator.mnist_data import mnist

import numpy as np
import tensorflow as tf
import random


def load_data(variable, variable_data, data_path, exp_num, get_data):
    """Load data."""

    if get_data:
        data = []
        for i in range(exp_num):
            data_load = np.load(data_path + 'data/exp{}.npz'.format(i))
            d = {variable[j]: data_load[variable_data[j]]
                 for j in range(len(variable))}
            data.append(d)
        return data
    else:
        return data_path


def gen_load_data(all_paras, get_data=False, autoiv_gen=False):
    """Generate and load data."""

    data_path = all_paras['data_path']
    exp_num = all_paras['exp_num']
    da = all_paras['dataset']

    if (da == 'sin') or (da == 'step') or (da == 'abs') or (da == 'linear') or (da == 'poly2d') or (da == 'poly3d'):
        data = 'toy'
    else:
        data = 'mnist'

    """If gen_flag is True or data is not exist, generate data; else load data."""
    if all_paras['gen_new'] or (not os.path.exists(data_path)):
        if data == 'toy':
            data_or_path = toy(all_paras, get_data)
        else:
            data_or_path = mnist(all_paras, get_data)
    else:
        if autoiv_gen:
            variable = ['v', 'z', 'c', 'x', 'y', 'ye', 'v_c0',
                        'z_c0', 'c_c0', 'exp_num', 'train', 'valid', 'test']
        else:
            variable = ['v', 'z', 'c', 'x', 'y', 'ye',
                        'exp_num', 'train', 'valid', 'test']

        data_or_path = load_data(
            variable, variable, data_path, exp_num, get_data=get_data)

    print('\nUse data: {}\n'.format(data_path))

    return data_or_path
