# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:25:35 2022

@author: chenming

"""


import numpy as np
from bert_whitening.modules import transform_and_normalize
from bert_whitening.hyperparameters import Hyperparamters as hp
from bert_whitening.modules import get_tokenizer,get_encoder
from bert_whitening.utils import load_train_data,load_test_data
from bert_whitening.modules import convert_to_vecs
from bert_whitening.modules import compute_kernel_bias
from bert_whitening.modules import compute_corrcoef



# 加载分词器 和 BERT向量器
tokenizer = get_tokenizer(hp.vocab_path)
encoder = get_encoder(hp.config_path, hp.checkpoint_path)


# =============================================================================
# 训练集：STS-B
# =============================================================================
# 加载数据集
datasets = {
    'sts-b-train': load_train_data('datasets/chn/senteval_cn/STS-B/STS-B.train.data'),
    'sts-b-test': load_test_data('datasets/chn/senteval_cn/STS-B/STS-B.test.data'),
    'sts-b-valid': load_test_data('datasets/chn/senteval_cn/STS-B/STS-B.valid.data')
}

# 语料向量化
all_names, all_weights, all_vecs, all_labels = [], [], [], []
all_query1, all_query2 = [], []
for name, data in datasets.items():
    a_vecs, b_vecs, labels, query1, query2 = convert_to_vecs(data, tokenizer, encoder)
    all_names.append(name)
    all_weights.append(len(data))
    all_vecs.append((a_vecs, b_vecs))
    all_labels.append(labels)
    all_query1.append(query1)
    all_query2.append(query2)

# 计算变换矩阵和偏置项
kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs])

# 变换，标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))
