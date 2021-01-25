'''
Author: lqyang
Date: 2021-01-22 16:13:23
LastEditTime: 2021-01-25 14:54:51
LastEditors: lqyang
Description: In User Settings Edit
FilePath: \session_related\FPMC-1\run.py
'''
import sys, os, pickle, argparse
from random import shuffle
from utils import *
from data2fpmc import load_data

from FPMC import FPMC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',help='dataset name, such as: tafeng_test/tafeng_0117', type=str, default='tafeng_test')
    parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=15)
    parser.add_argument('--n_neg', help='# of neg samples', type=int, default=10)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=32)
    parser.add_argument('-l', '--learn_rate', help='learning rate', type=float, default=0.01)
    parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.001)
    args = parser.parse_args()
    print(args)

    datasets_dir = os.path.join('datasets', args.dataset)

    tr_data, te_data, item_set = load_data(datasets_dir) 
    # tr_data: list, [(label, [item_index0, item_index1, ...]]), ...]
    # te_data: list, [(label, [item_index0, item_index1, ...]]), ...]
    # item_set: set, {item_index, ...}


    fpmc = FPMC(n_item=max(item_set)+1, 
                n_factor=args.n_factor, learn_rate=args.learn_rate, regular=args.regular)

    fpmc.item_set = item_set
    fpmc.init_model()

    acc, mrr = fpmc.learnSBPR_FPMC(tr_data, te_data, n_epoch=args.n_epoch, 
                                   neg_batch_size=args.n_neg, eval_per_epoch=True)

    print ("Accuracy:%.2f MRR:%.2f" % (acc, mrr))






