'''
Author: lqyang
Date: 2021-01-25 13:30:45
LastEditors: lqyang
LastEditTime: 2021-01-25 14:41:21
FilePath: \session_related\FPMC-1\data2fpmc.py
Dec: 
    Convert to FPMC data format
    Namely:
        # tra_data_list: list, [(label, [item_index0, item_index1, ...]]), ...]
        # tes_data_list: list, [(label, [item_index0, item_index1, ...]]), ...]
        # item_set: set, {item_index, ...}
'''

import os
import json
import pickle
import numpy as np

def load_data(dataset_dir):
    '''
    description: 
    1) Loads the dataset
    param {dataset_dir: str}: eg.'datasets/tb_sample'
    return {*}:
        tra_data_list: list, [(label, [item_index0, item_index1, ...]]), ...]
        tes_data_list: list, [(label, [item_index0, item_index1, ...]]), ...]
        item_set: set, {item_index, ...}
    '''
    # Load the dataset
    path_train_data = os.path.join(dataset_dir, 'train.txt')
    path_test_data = os.path.join(dataset_dir, 'test.txt')
    session_info_path = os.path.join(dataset_dir, 'session_info.json')
    with open(session_info_path, 'r') as f:
        session_info = json.load(f)
    with open(path_train_data, 'rb') as f1:
        train_set = pickle.load(f1)
    with open(path_test_data, 'rb') as f2:
        test_set = pickle.load(f2)

    tra_data_list = [tuple((label, b_tem1)) for label, b_tem1 in zip(train_set[1], train_set[0])]
    tes_data_list = [tuple((label, b_tem1)) for label, b_tem1 in zip(test_set[1], test_set[0])]

    try:
        num_items = session_info["num_of_item[not include 0]"] + 1
    except:
        num_items = session_info["num_of_item"] + 1
    item_set = set(np.arange(num_items))

    return tra_data_list, tes_data_list, item_set

    
if __name__ == "__main__":
    dataset_dir = os.path.join("datasets", "tafeng_test")
    print(dataset_dir)
    tra_data_list, tes_data_list, item_set = load_data(dataset_dir)

    


