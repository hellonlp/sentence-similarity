# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:53:30 2023

@author: Chen Ming
"""


import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
print('pwd:', pwd)

import heapq
import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from simcse.utils import load_data_new
from simcse.hyperparameters import Hyperparameters as hp
from sklearn.metrics.pairwise import cosine_similarity
from simcse.networks import SimCSEModelSup



# parameters
pretrained_model_path = hp.pretrained_model_path
simcse_path = hp.LOAD_PATH_SUP # ***
DEVICE = hp.DEVICE 
MAXLEN = hp.MAXLEN 
BATCH_SIZE = hp.BATCH_SIZE
POOLING = hp.POOLING 

# model
tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
MODEL = SimCSEModelSup(pretrained_model=pretrained_model_path, pooling=hp.POOLING)
MODEL.load_state_dict(torch.load(simcse_path))





class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text):
        return tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        da = self.data[index]        
        return self.text_2_id([da[0]]), self.text_2_id([da[1]]), int(da[2])



class InferenceDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text):
        return tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        da = self.data[index]        
        return self.text_2_id([da])


def get_vector_simcse(sentence, model=MODEL):
    """
    预测simcse的语义向量。
    """
    data_feed = list(DataLoader(InferenceDataset([sentence]), batch_size=1))[0]
    model.eval()
    with torch.no_grad():
        input_ids = data_feed.get('input_ids').squeeze(1).to(DEVICE)
        attention_mask = data_feed.get('attention_mask').squeeze(1).to(DEVICE)
        token_type_ids = data_feed.get('token_type_ids').squeeze(1).to(DEVICE)
        output_vector = model(input_ids, attention_mask, token_type_ids)
    return output_vector#[0]



def get_similarity_topN_new(sentence, texts, topN=10):  
    vec = get_vector_simcse(sentence)
    texts_vec = [get_vector_simcse(text) for text in texts]
    similarity_list = [cosine_similarity(vec, l)[0] for l in texts_vec]
    topk_index = heapq.nlargest(topN, range(len(similarity_list)), similarity_list.__getitem__)
    outputs = []
    for i in topk_index:
        outputs.append((np.round(similarity_list[i],4)[0],texts[i]))
    return outputs    


def get_similarity_two(sentence1, sentence2, topN=10):  
    vec1 = get_vector_simcse(sentence1)
    vec2 = get_vector_simcse(sentence2)
    similarity_list = cosine_similarity(vec1, vec2).tolist()[0][0]
    return similarity_list



if __name__ == '__main__':
    # 加载数据集
    datasets_sts = {
        'STS-B-train': load_data_new('F:/celery/datasets/chn/senteval_cn/STS-B/STS-B.train.data'),
        'STS-B-test': load_data_new('F:/celery/datasets/chn/senteval_cn/STS-B/STS-B.test.data'),
        'STS-B-valid': load_data_new('F:/celery/datasets/chn/senteval_cn/STS-B/STS-B.valid.data')
    }

    #
    import time
    start = time.time()
    sentence = '一个小孩在骑马。' #[datasets_sts['STS-B-valid'][1][0]]
    vector = get_vector_simcse(sentence)#[0]
    print(vector.shape)
    end = time.time()
    print('Time cost:', end - start)
    
    
    
    # 和新数据对比
    texts = ['你怎么样','我吃了一个苹果','你过的好吗','你还好吗','你',
             '你好不好','你好不好呢','我不开心','我好开心啊', '你吃饭了吗',
             '你好吗','你现在好吗','你好个鬼']
    sentence = '你好吗'
    results = get_similarity_topN_new(sentence,texts, topN=20)
    for l in results:
        print(l)   
    
    # 两个句子之间
    sentence1 = '你好吗'
    sentence2 = '你过的好吗'
    print(get_similarity_two(sentence1,sentence2))
    

# (1.0, '你好吗')
# (0.8247, '你好不好')
# (0.8217, '你现在好吗')
# (0.7976, '你还好吗')
# (0.7918, '你好不好呢')
# (0.712, '你过的好吗')
# (0.6986, '你怎么样')
# (0.6693, '你')
# (0.5442, '你好个鬼')
# (0.4516, '你吃饭了吗')
# (0.29, '我不开心')
# (0.2782, '我吃了一个苹果')




   