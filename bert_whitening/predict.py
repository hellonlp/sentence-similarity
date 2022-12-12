# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:44:40 2022

@author: chenming
"""


import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bert_whitening.utils import load_model,load_vectors
from bert_whitening.hyperparameters import Hyperparamters as hp
from bert_whitening.modules import get_tokenizer,get_encoder
from bert_whitening.modules import convert_sentence_to_vecs
from bert_whitening.modules import transform_and_normalize
from bert_whitening.modules import convert_sentences_to_vecs


# 加载模型 和 加载向量
kernel, bias = load_model(hp.model_load_path)
vectors, querys = load_vectors(hp.vectors_path)

# 加载分词器 和 BERT向量器
tokenizer = get_tokenizer(hp.vocab_path)
encoder = get_encoder(hp.config_path, hp.checkpoint_path)



def get_similarity_topN(sentence, topN=10):
    """
    Parameters
    ----------
    sentence : TYPE string
        DESCRIPTION. A sentence.
    topN : TYPE int
        DESCRIPTION. The default is 5.

    Returns
    -------
    outputs : a list of tuple
        DESCRIPTION. The result tuples like:
            (0.5286, '一架飞机正在起飞。')
            (0.5283, '有些人在战斗。')
            (0.5271, '一架飞机要起飞了。')
            (0.5265, '一个人在切黄瓜。')
            (0.5265, '一个女孩在给她的头发做发型。')
    """
    vec = convert_sentence_to_vecs(sentence, tokenizer, encoder)
    trans_vec = transform_and_normalize(vec, kernel, bias)
    similarity_list = cosine_similarity(trans_vec, vectors).tolist()[0]
    topk_index = heapq.nlargest(topN, range(len(similarity_list)), similarity_list.__getitem__)
    outputs = []
    for i in topk_index:
        outputs.append((np.round(similarity_list[i],4),querys[i]))
    return outputs

    
def get_similarity_topN_new(sentence, texts, topN=10):  
    vec = convert_sentence_to_vecs(sentence, tokenizer, encoder)
    trans_vec = transform_and_normalize(vec, kernel, bias)
    texts_vec = convert_sentences_to_vecs(texts, tokenizer, encoder)
    texts_vec_trans = transform_and_normalize(texts_vec, kernel, bias)
    similarity_list = cosine_similarity(trans_vec, texts_vec_trans).tolist()[0]
    topk_index = heapq.nlargest(topN, range(len(similarity_list)), similarity_list.__getitem__)
    outputs = []
    for i in topk_index:
        outputs.append((np.round(similarity_list[i],4),texts[i]))
    return outputs    


def get_similarity_two(sentence1, sentence2, topN=10):  
    vec = convert_sentence_to_vecs(sentence1, tokenizer, encoder)
    trans_vec = transform_and_normalize(vec, kernel, bias)
    texts_vec = convert_sentences_to_vecs([sentence2], tokenizer, encoder)
    texts_vec_trans = transform_and_normalize(texts_vec, kernel, bias)
    similarity_list = cosine_similarity(trans_vec, texts_vec_trans).tolist()[0][0]
    return similarity_list


if __name__ == '__main__':    
    # 和历史数据对比
    sentence = '一架飞机正在起飞。'
    results = get_similarity_topN(sentence)
    for l in results:
        print(l)
    
    # 和新数据对比
    texts = ['你怎么样','我吃了一个苹果','你过的好吗','你还好吗','你',
             '你好不好','你好不好呢','我不开心']
    sentence = '你好吗'
    results = get_similarity_topN_new(sentence,texts)
    for l in results:
        print(l)   
    
    #
    sentence1 = '你好吗'
    sentence2 = '你过的好吗'
    print(get_similarity_two(sentence1,sentence2))
    
