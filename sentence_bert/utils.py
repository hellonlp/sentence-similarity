# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:37:43 2022

@author: chenming
"""


import numpy as np


def load_sts_b_train_data(filename):
    """加载STS-B训练数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('\t')
                D.append((l[-3], l[-2], float(l[-1])))
    return D


def load_sts_b_test_data(filename):
    """加载STS-B测试数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            D.append((l[-2], l[-1], float(l[-3])))
    return D


def load_sts_12_16_data(filename):
    """加载STS-12,13,14,15,16数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    input_file = filename
    label_file = input_file.replace('STS.input', 'STS.gs')
    input_file = open(input_file, encoding='utf-8')
    label_file = open(label_file, encoding='utf-8')
    for i, l in zip(input_file, label_file):
        if l.strip():
            i = i.strip().split('\t')
            l = float(l.strip())
            D.append((i[0], i[1], l))
    input_file.close()
    label_file.close()
    return D


def load_sick_r_data(filename):
    """加载SICK-R数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('\t')
                D.append((l[1], l[2], float(l[3])))
    return D


def load_mnli_train_data(filename):
    """加载MNLI训练数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('\t')
                D.append((l[8], l[9], l[10]))
    return D


def load_lcqmc_train_data(filename):
    """加载训练数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = l.strip().split('\t')
            try:
                D.append((l[0], l[1], float(l[2])))
            except:
                D = D
    return D


def load_snli_data(filename):
    """加载SNLI数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    filename = filename.split('/')
    s1_file = '/'.join(filename[:-1]) + '/s1.' + filename[-1]
    s2_file = '/'.join(filename[:-1]) + '/s2.' + filename[-1]
    l_file = '/'.join(filename[:-1]) + '/labels.' + filename[-1]
    s1_file = open(s1_file, encoding='utf-8')
    s2_file = open(s2_file, encoding='utf-8')
    l_file = open(l_file, encoding='utf-8')
    for s1, s2, l in zip(s1_file, s2_file, l_file):
        D.append((s1.strip(), s2.strip(), l.strip()))
    s1_file.close()
    s2_file.close()
    l_file.close()
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
    
