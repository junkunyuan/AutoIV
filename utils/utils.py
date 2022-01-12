import datetime
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
import csv


def create_path(path):
    """Create path like: 'a/b/c/'."""
    path_split = path.split('/')
    temp = path_split[0] + '/'
    for i in range(1, len(path_split)):
        if not os.path.exists(temp):
            os.mkdir(temp)
        temp = temp + path_split[i] + '/'


def get_line(x, y, x_int_times=50, x_min=-5, x_max=5):
    interval = (x_max - x_min) / x_int_times
    x_new, y_new = [], []
    for int_i in range(x_int_times):
        start = x_min + interval * int_i
        end = x_min + interval * (int_i + 1)
        get_data = np.where((x > start) & (x < end))
        x_new = x_new + [(start + end) / 2]
        y_new = y_new + [np.mean(y[get_data])]
    return np.array(x_new), np.array(y_new)


class Log(object):
    def __init__(self, all_paras):
        """Result log."""

        self.coefs = all_paras['coefs']
        self.date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.res_path = all_paras['res_path']
        self.res_detail_path = self.res_path + \
            '{}-train_{}-'.format(all_paras['dataset'],
                                  all_paras['train']) + self.date + '/'

        self.res_summary_file = self.res_path + 'result-sum.txt'
        self.res_detail_file = self.res_detail_path + 'result-det.txt'

        self.all_paras = all_paras

        """Delete and create directory."""
        self.del_cre_directory()

    def del_cre_directory(self):
        """Delete previous result and create directory."""

        """If remove all the detail results."""
        if self.all_paras['del_res'] and os.path.exists(self.res_path):
            shutil.rmtree(self.res_path)

        """Remove summary results."""
        if os.path.exists(self.res_summary_file):
            os.remove(self.res_summary_file)

        """Create directory."""
        create_path(self.res_detail_path)

        self.write('both', self.date)

    def write(self, _file, _str, _print_flag=False):
        """Write str in summary file, or detail file, or both."""

        if _print_flag:
            print(_str)
        if _file == 'summary':
            with open(self.res_summary_file, 'a') as f:
                f.write(_str + '\n')
        elif _file == 'detail':
            with open(self.res_detail_file, 'a') as f:
                f.write(_str + '\n')
        elif _file == 'both':
            with open(self.res_summary_file, 'a') as f:
                f.write(_str + '\n')
            with open(self.res_detail_file, 'a') as f:
                f.write(_str + '\n')
        else:
            raise Exception('Wrong value for writing log file!')

    def get_path(self, _file):
        """Get path of summary path or detail path."""

        if _file == 'summary':
            return self.res_path
        elif _file == 'detail':
            return self.res_detail_path
        else:
            raise Exception('Wrong value for getting file path!')


def exp_log(sta_end_flag, log, all_paras):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if sta_end_flag == 'start':
        log.write('both', '=' * 50)
        log.write('both', 'Start training: ' + time_now)
        log.write('both', 'data_train: {}; data_valid: {}; data_test: {}'.format(
            all_paras['train'], all_paras['valid'], all_paras['test']))
        """ Log hyper-parameters. """
        log.write('both', '-' * 50)
        log.write('both', 'rep_net_layer: {}'.format(
            all_paras['rep_net_layer']))
        log.write('both', 'x_net_layer: {}'.format(all_paras['x_net_layer']))
        log.write('both', 'emb_net_layer: {}'.format(
            all_paras['emb_net_layer']))
        log.write('both', 'y_net_layer: {}'.format(all_paras['y_net_layer']))
        log.write('both', 'emb_dim: {}'.format(all_paras['emb_dim']))
        log.write('both', 'rep_dim: {}'.format(all_paras['rep_dim']))
        log.write('both', 'learning rate: {}'.format(all_paras['lrate']))
        log.write('both', 'dropout: {}'.format(all_paras['dropout']))
        log.write('both', 'epochs: {}'.format(all_paras['epochs']))
        log.write('both', 'opt_lld_step: {}'.format(all_paras['opt_lld_step']))
        log.write('both', 'opt_bound_step: {}'.format(
            all_paras['opt_bound_step']))
        log.write('both', 'opt_2stage_step: {}'.format(
            all_paras['opt_2stage_step']))
        log.write('both', 'sigma: {}'.format(all_paras['sigma']))
        for key in all_paras['coefs']:
            log.write('both', key + ': ' + str(all_paras['coefs'][key]))
        log.write('both', '-' * 50)

    elif sta_end_flag == 'end':
        log.write('both', 'Finish training: ' + time_now)
        log.write('both', '=' * 50)


def res_val_save(model, all_para, dicts, exp):
    """Save variables after training."""

    """Get feed_dict."""
    dict_all, dict_all_c0 = dicts[0], dicts[1]

    z, c, x, y, x_pre, y_pre = model.sess.run(
        [model.rep_z, model.rep_c, model.x, model.y, model.x_pre, model.y_pre], feed_dict=dict_all)
    z_c0, c_c0 = model.sess.run(
        [model.rep_z, model.rep_c], feed_dict=dict_all_c0)

    """Data path."""
    path_split = all_para['data_path'].split('/')
    gen_dciv = 'autoiv-' + all_para['dataset']
    data_path = path_split[0] + '/' + path_split[1] + '/' + gen_dciv + '/' + \
        gen_dciv + \
        '-train_{}-rep_{}/data/'.format(all_para['train'], all_para['rep_dim'])
    create_path(data_path)

    """Save data in csv."""
    num = all_para['train'] + all_para['valid'] + all_para['test']
    with open(data_path + 'exp{}.csv'.format(exp), 'w', newline='') as f:
        f.write('x,x_pre, y, y_pre\n')
        csv_writer = csv.writer(f, delimiter=',')
        for j in range(num):
            temp = [x[j][0], x_pre[j][0], y[j][0], y_pre[j][0]]
            temp.extend(z[j, :])
            temp.extend(c[j, :])
            csv_writer.writerow(temp)

    """Save data in npz."""
    v = np.concatenate([z, c], axis=1)
    v_c0 = np.concatenate([z_c0, c_c0], axis=1)
    np.savez(data_path + 'exp{}.npz'.format(exp),
             v=v, z=z, c=c, x=x, y=y, ye=y, v_c0=v_c0, z_c0=z_c0, c_c0=c_c0,
             train=all_para['train'], valid=all_para['valid'], test=all_para['test'], exp_num=all_para['exp_num'])

    print(data_path + 'exp{}.npz'.format(exp))
