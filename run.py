'''
Author: lqyang
Date: 2021-01-22 16:13:23
LastEditTime: 2021-01-25 17:00:32
LastEditors: lqyang
Description: In User Settings Edit
FilePath: \session_related\FPMC-1\run.py
'''
import sys, os, pickle, argparse
from random import shuffle
from utils import *
from data2fpmc import load_data
from utils_add import *
from FPMC import FPMC


import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',help='dataset name, such as: tafeng_test/tafeng_0117', type=str, default='tafeng_test')
    parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=15)
    parser.add_argument('--n_neg', help='# of neg samples', type=int, default=10)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=32)
    parser.add_argument('-l', '--learn_rate', help='learning rate', type=float, default=0.01)
    parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.001)
    parser.add_argument('--log', '--log_path_txt', help='log_path_txt', type=str, default="")
    args = parser.parse_args()
    print(args)

    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))
    log_dir_train = os.path.join('./visual', args.dataset, 'DemandRS',
                                 f"{time_path}_lr{args.learn_rate}")
    log_dir_train_checkpoint = os.path.join(log_dir_train, "checkpoint")
    log_path_txt = os.path.join(log_dir_train, "output.txt")
    args.log_path_txt = log_path_txt

    if not os.path.exists(log_dir_train):
        os.makedirs(log_dir_train)
    if not os.path.exists(log_dir_train_checkpoint):
        os.makedirs(log_dir_train_checkpoint)
    
    seed_torch(args.log_path_txt, seed=2021)
    datasets_dir = os.path.join('datasets', args.dataset)

    tr_data, te_data, item_set = load_data(datasets_dir) 
    # tr_data: list, [(label, [item_index0, item_index1, ...]]), ...]
    # te_data: list, [(label, [item_index0, item_index1, ...]]), ...]
    # item_set: set, {item_index, ...}


    fpmc = FPMC(n_item=max(item_set)+1, 
                n_factor=args.n_factor, learn_rate=args.learn_rate, regular=args.regular)

    fpmc.item_set = item_set
    fpmc.init_model()

    Ks = [10,20,40,50,60,80,100]
    Ks_auc = [50,100,200,500]

    start = time.time()
    acc, mrr = fpmc.learnSBPR_FPMC(args.log_path_txt, tr_data, te_data, n_epoch=args.n_epoch, 
                                   neg_batch_size=args.n_neg, eval_per_epoch=True, Ks=Ks, Ks_auc=Ks_auc)
    file_write(args.log_path_txt, '-------------------------------------------------------')
    end = time.time()
    file_write(log_path_txt, f"Run time: {(end - start) / 60.0}min")


    






