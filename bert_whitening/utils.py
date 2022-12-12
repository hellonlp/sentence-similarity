# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:57:48 2022

@author: chenming
"""


import numpy as np



def load_train_data(filename):
    """加载训练数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            #if i > 0:
            l = l.strip().split('\t')
            try:
                D.append((l[0], l[1], float(l[2])))
            except:
                D = D
    return D


def load_test_data(filename):
    """加载测试数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            D.append((l[0], l[1], float(l[2])))
    return D


def save_model(file_save_model, kernel, biais):
    np.savez(file_save_model, kernel, biais)
    print('Save model finished!')


def load_model(file_load_model):
    model = np.load(file_load_model)
    print('Load model finished!')
    return model['arr_0'], model['arr_1']


def save_vectors(file_save_vector, vectors, querys):
    np.savez(file_save_vector, vectors, querys)
    print('Save vector finished!')


def load_vectors(file_load_vector):
    model = np.load(file_load_vector)
    print('Load vector finished!')
    return model['arr_0'], model['arr_1']
    
    
