# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:41:29 2022

@author: chenming
"""



import numpy as np
from bert_whitening.utils import load_train_data,load_test_data
from bert_whitening.modules import get_tokenizer,get_encoder
from bert_whitening.hyperparameters import Hyperparamters as hp
from bert_whitening.modules import transform_and_normalize,compute_corrcoef
from bert_whitening.modules import convert_to_vecs,compute_kernel_bias
from bert_whitening.utils import save_model,save_vectors




# 加载数据集
datasets = {
    'sts-b-train': load_train_data('datasets/STS-B/STS-B.train.data'),
    'sts-b-test': load_test_data('datasets/STS-B/STS-B.test.data')
}

# 加载分词器 和 BERT向量器
tokenizer = get_tokenizer(hp.vocab_path)
encoder = get_encoder(hp.config_path, hp.checkpoint_path)

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
all_corrcoefs,all_vecs_trans = [],[]
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)
    all_vecs_trans.append(a_vecs)
    all_vecs_trans.append(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)
    
# 计算相关性
all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])
for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))
    
    
    
# 向量
vecs_new = np.concatenate(all_vecs_trans, axis=0)

# 句子
querys_new = np.concatenate([all_query1[0],all_query2[0],all_query1[1],all_query2[1]], axis=0)

# 去重
query_vec_dict = {}
for i in range(len(querys_new)):
    q = querys_new[i]
    if q not in query_vec_dict.keys():
        query_vec_dict[q] = vecs_new[i]
vecs_new_save, querys_new_save = [], []
for k, v in query_vec_dict.items():
    querys_new_save.append(k)
    vecs_new_save.append(v)
    

# 本地保存 kernel, bias
f_model = 'model/V1.0/model_STS-B.npz'
save_model(f_model,kernel,bias)  
f_vector = 'model/V1.0/vector_STS-B.npz'
save_vectors(f_vector, vecs_new_save, querys_new_save)


    
    
