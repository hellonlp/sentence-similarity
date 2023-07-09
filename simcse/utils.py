# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:57:11 2023

@author: Chen Ming
"""


import jsonlines


def load_data_new(filename):
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


def load_snli_data(path):        
    with jsonlines.open(path, 'r') as f:
        return [(line['origin'], line['entailment'], line['contradiction']) for line in f]
        