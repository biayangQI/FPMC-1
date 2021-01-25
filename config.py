'''
Author: lqyang
Date: 2021-01-25 20:03:19
LastEditors: lqyang
LastEditTime: 2021-01-25 22:11:00
FilePath: \FPMC-1\config.py
Dec: 
'''
import warnings
import torch as t
from utils_add import file_write

class DefaultConfig(object):
    beizhu = 'server: 163@10.249.42.112, tafeng_test for debug '  #'server: 163@10.249.182.163

    dataset = "tafeng_test" # help='dataset name, such as: tafeng_test/tafeng_0117', type=str, default='tafeng_test'
    n_epoch = 15 # help='# of epoch', type=int, default=1000
    n_neg = 10 # help='# of neg samples', type=int, default=10
    n_factor = 32 # help='dimension of factorization', type=int, default=32
    learn_rate = 0.01 # help='learning rate', type=float, default=0.01
    regular = 0.001 # help='regularization', type=float, default=0.001
    log_path_txt = "" #  help='log_path_txt', type=str, default=""
    try:
        file_write(log_path_txt, '------------import config.py---------------')
    except:
        print("")

    def parse(self):
        """
        根据字典kwargs 更新 config参数
        """
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')
        """
        file_write(self.log_path_txt, 'user config:')
        parameters= []
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_') and k not in ('log_path_txt'):
                parameters.append((k, getattr(self, k)))
        parameters_str = ''.join(str(e) for e in parameters)
        file_write(self.log_path_txt, parameters_str)

args = DefaultConfig()


"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',help='dataset name, such as: tafeng_test/tafeng_0117', type=str, default='tafeng_test')
    parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=15)
    parser.add_argument('--n_neg', help='# of neg samples', type=int, default=10)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=32)
    parser.add_argument('-l', '--learn_rate', help='learning rate', type=float, default=0.01)
    parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.001)
    parser.add_argument('--log', '--log_path_txt', help='log_path_txt', type=str, default="")
    args = parser.parse_args()
"""
