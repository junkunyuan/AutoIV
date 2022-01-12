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


def toy(all_paras, get_data):
    """Generate toy (low-dimensinal scenarios) data."""

    """Set paras."""
    num = all_paras['train'] + all_paras['valid'] + all_paras['test']
    exp_num = all_paras['exp_num']

    """Create fold for data."""
    create_path(all_paras['data_path'])

    # A folder for saving information.
    info_path = all_paras['data_path'] + 'info/'
    if not os.path.exists(info_path):
        os.mkdir(info_path)

    # A folder for saving csv and npz data.
    data_path = all_paras['data_path'] + 'data/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    datas = []
    print('=' * 50 + '\nStart to generate new data.')
    for exp_i in range(exp_num):
        """Generate Z ~ Uniform([-3, 3]^2)."""
        z = np.random.uniform(low=-3., high=3., size=(num, 2))

        """Generate e ~ N(0, 1)."""
        e = np.random.normal(0., 1., size=(num, 1))

        """Generate gamma ~ N(0, 0.1)."""
        gamma = np.random.normal(0., .1, size=(num, 2))

        """Generate x = z1 + e + gamma."""
        two_gps = False
        iv_strength = 0.5
        if two_gps:
            x = 2 * z[:, 0:1] * (z[:, 0:1] > 0) * iv_strength + 2 * z[:, 1:2] * (z[:, 1] < 0) * iv_strength + \
                2 * e * (1 - iv_strength) + gamma[:, 0:1]
        else:
            x = 2 * z[:, 0:1] * iv_strength + 2 * \
                e * (1 - iv_strength) + gamma[:, 0:1]

        """Generate y = g0(x) + e + delta."""
        if all_paras['dataset'] == 'sin':
            g0 = np.sin(x)
        elif all_paras['dataset'] == 'step':
            g0 = - np.heaviside(x, 1.0)
        elif all_paras['dataset'] == 'abs':
            g0 = np.abs(x)
        elif all_paras['dataset'] == 'linear':
            g0 = -1.0 * x
        elif all_paras['dataset'] == 'poly2d':
            g0 = -0.4 * x - 0.1 * (x**2)
        elif all_paras['dataset'] == 'poly3d':
            g0 = -0.8 * x + 0.1 * (x**2) + 0.05 * (x ** 3)

        y = g0
        ye = g0 + gamma[:, 1:2] + 2. * e

        y = (y - np.mean(y)) / np.std(y)
        ye = (ye - np.mean(ye)) / np.std(ye)

        """Draw pictures."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_draw, y_draw = get_line(x, y)
        ax.plot(x_draw, y_draw, label='structural function',
                linewidth=5, c='royalblue', zorder=10)
        ax.scatter(x, ye, label='data', s=6, c='limegreen', zorder=1)
        plt.xlim(-4, 4)
        plt.ylim(-3, 3)
        plt.savefig(info_path + 'y_{}-train_{}-distribution{}.png'
                    .format(all_paras['dataset'], all_paras['train'], exp_i + 1))
        plt.close(fig)

        """Write data in csv."""
        # Write data distribution.
        with open(info_path + 'x_y_ye-train_{}-distribution_{}.txt'.format(all_paras['train'], exp_i + 1), 'w') as f:
            f.write('\nmean_std_x: {}, {}\n'.format(np.mean(x), np.std(x)))
            f.write('\nmedian_min_max_x: {}, {}, {}\n'.format(
                np.median(x), np.min(x), np.max(x)))
            f.write('\nmean_std_y: {}, {}\n'.format(np.mean(y), np.std(y)))
            f.write('\nmedian_min_max_y: {}, {}, {}\n'.format(
                np.median(y), np.min(y), np.max(y)))
            f.write('\nmean_std_ye: {}, {}'.format(np.mean(ye), np.std(ye)))
            f.write('\nmedian_min_max_ye: {}, {}, {}\n'.format(
                np.median(ye), np.min(ye), np.max(ye)))

        # Write data.
        with open(data_path + 'exp{}.csv'.format(exp_i), 'w', newline='') as f:
            f.write('x, y, ye\n')
            csv_writer = csv.writer(f, delimiter=',')
            for j in range(num):
                temp = [x[j][0], y[j][0], ye[j][0]]
                csv_writer.writerow(temp)

        v = np.concatenate([z, gamma], axis=1)

        """Write data in npz."""
        np.savez(data_path + 'exp{}.npz'.format(exp_i), v=v, z=z, c=gamma, x=x, y=y, ye=ye,
                 train=all_paras['train'], valid=all_paras['valid'], test=all_paras['test'], exp_num=exp_num)

        if get_data:
            datas = datas + [{'v': v, 'z': z, 'c': gamma, 'x': x, 'y': y, 'ye': ye, 'exp_num': exp_num,
                              'train': all_paras['train'], 'valid': all_paras['valid'], 'test': all_paras['test']}]

    print('Data size:\n\tz: {}\n\tc: {}\n\tx: {}\n\ty: {}'.format(
        z.shape, gamma.shape, x.shape, y.shape))
    print('Finish generating and saving data.\n' + '=' * 50)

    """Return data or path."""
    if get_data:
        return datas
    else:
        return all_paras['data_path']
