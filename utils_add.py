#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: lqyang
Date: 2021-01-25 16:08:10
LastEditors: lqyang
LastEditTime: 2021-01-25 17:04:21
Dec: 
    some useful functions
'''


import time
import os
import psutil
import torch
from itertools import chain
import numpy as np
import random
from functools import wraps

import torch

#  fix seed
def seed_torch(log_path_txt, seed=None):
    """
    Args:
        log_path_txt: 文件路径，such as： output.txt
        seed:
    Returns: None
    """

    if seed is None:
        seed = int(time.time())
    file_write(log_path_txt, f'************ seed ***********: {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子


def print_result(log_file, title,metric, Ks=[10,20,50]):
    """
    Args:
        log_file: 文件路径，such as： output.txt
        title: str, 'recall', 'mrr', 'ndcg'
        metric: list, len=len(Ks)
    """
    assert len(metric)==len(Ks)
    title_str = '%s@%d\t' * len(Ks)
    file_write(log_file, title_str %(title, Ks[0], title, Ks[1],title, Ks[2],title, Ks[3],title, Ks[4], title, Ks[5],
       title, Ks[6]))
    #file_write(log_file, f"{metric[0]}\t{metric[1]}\t{metric[2]}\t{metric[3]}\t{metric[4]}\t{metric[5]}\t{metric[6]}\t")
    #"""
    result_str = '%.4f\t' * len(Ks)
    file_write(log_file, result_str % (metric[0], metric[1], metric[2], metric[3],metric[4],
        metric[5],metric[6]))
    #"""
    return


def print_auc_result(log_file,title, metric, Ks_auc=[50,100,200,500]):
    """
    Args:
        log_file: 文件路径，such as： output.txt
        title: str, 'auc'
        metric: list, len=len(Ks_auc)
    """
    assert len(metric) == len(Ks_auc)
    """
    file_write(log_file, f'{title}@{Ks_auc[0]}\t{title}@{Ks_auc[1]}\t{title}@{Ks_auc[2]}\t{title}@{Ks_auc[3]}')
    file_write(log_file, f"{metric[0]}\t{metric[1]}\t{metric[2]}\t{metric[3]}")
    """
    #"""
    title_str = '%s@%d\t' * len(Ks_auc)
    file_write(log_file, title_str % (title, Ks_auc[0], title, Ks_auc[1], title, Ks_auc[2], title, Ks_auc[3]))
    result_str = '%.4f\t' * len(Ks_auc)
    file_write(log_file, result_str % (metric[0], metric[1], metric[2], metric[3]))
    #"""
    return


def cprint(log_file, words: str):
    """
    Highlight words
    Args:
        log_file: 文件路径，such as： output.txt
        words: str
    Returns: None
    """
    print(f"\033[31;1m{words}\033[0m")
    file_write(log_file, '\n', whether_print=False)
    file_write(log_file,words, whether_print=False)


def file_write(log_file, s, whether_print=True):
    """
    将字符 s 写入文件 log_file, 以txt形式，非二进制文件
    Args:
        log_file: 文件路径，such as： output.txt
        s: str, 要写入文件的内容
        whether_print: bool， default=True， 是否打印写入的字符串内容
    """

    if whether_print:
        print(s)
    with open(log_file, 'a') as f:  # 'a' 表示打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
        f.write(s+'\n')
