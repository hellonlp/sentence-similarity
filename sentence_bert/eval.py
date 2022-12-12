# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 17:19:17 2022

@author: chenming
"""



import numpy as np
from sentence_bert.modules import transform_and_normalize
from sentence_bert.hyperparameters import Hyperparamters as hp
from sentence_bert.modules import get_tokenizer,get_encoder
from sentence_bert.utils import load_lcqmc_train_data,load_model
from sentence_bert.modules import convert_to_vecs
from sentence_bert.modules import compute_kernel_bias
from sentence_bert.modules import compute_corrcoef



# 加载分词器 和 BERT向量器
tokenizer = get_tokenizer(hp.dict_path)
encoder = get_encoder(hp.config_path, hp.checkpoint_path)


# 加载LQCMC预训练权重
if hp.use_weigths:
    encoder.load_weights(hp.weights_load_path)
    
    
# =============================================================================
# 训练集：LQCMC
# =============================================================================
# 加载数据集
# datasets = {
#     'LCQMC-train': load_lcqmc_train_data('datasets/chn/senteval_cn/LCQMC/LCQMC.train.data'),
#     'LCQMC-test': load_lcqmc_train_data('datasets/chn/senteval_cn/LCQMC/LCQMC.test.data'),
#     'LCQMC-valid': load_lcqmc_train_data('datasets/chn/senteval_cn/LCQMC/LCQMC.valid.data')
# }

datasets = {
    'STS-B-train': load_lcqmc_train_data('datasets/chn/senteval_cn/STS-B/STS-B.train.data'),
    'STS-B-test': load_lcqmc_train_data('datasets/chn/senteval_cn/STS-B/STS-B.test.data'),
    'STS-B-valid': load_lcqmc_train_data('datasets/chn/senteval_cn/STS-B/STS-B.valid.data')
}



# 语料向量化
all_names, all_weights, all_vecs, all_labels = [], [], [], []
all_query1, all_query2 = [], []
for name, data in datasets.items():
    a_vecs, b_vecs, labels = convert_to_vecs(data, tokenizer, encoder)
    all_names.append(name)
    all_weights.append(len(data))
    all_vecs.append((a_vecs, b_vecs))
    all_labels.append(labels)


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
     
