# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:46:02 2022

@author: chenming
"""



import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_bert.hyperparameters import Hyperparamters as hp
from sentence_bert.modules import get_tokenizer,get_encoder
from sentence_bert.modules import convert_sentence_to_vecs
from sentence_bert.modules import convert_sentences_to_vecs



# 加载分词器 和 BERT向量器
tokenizer = get_tokenizer(hp.dict_path)
encoder = get_encoder(hp.config_path, hp.checkpoint_path)


# 加载LQCMC预训练权重
if hp.use_weigths:
    encoder.load_weights(hp.weights_load_path)

    
def get_similarity_topN_new(sentence, texts, topN=10):  
    vec = convert_sentence_to_vecs(sentence, tokenizer, encoder)
    texts_vec = convert_sentences_to_vecs(texts, tokenizer, encoder)
    similarity_list = cosine_similarity(vec, texts_vec).tolist()[0]
    topk_index = heapq.nlargest(topN, range(len(similarity_list)), similarity_list.__getitem__)
    outputs = []
    for i in topk_index:
        outputs.append((np.round(similarity_list[i],4),texts[i]))
    return outputs    


def get_similarity_two(sentence1, sentence2, topN=10):  
    vec = convert_sentence_to_vecs(sentence1, tokenizer, encoder)
    texts_vec = convert_sentences_to_vecs([sentence2], tokenizer, encoder)
    similarity_list = cosine_similarity(vec, texts_vec).tolist()[0][0]
    return similarity_list


if __name__ == '__main__':      
    # 和新数据对比
    texts = ['你怎么样','我吃了一个苹果','你过的好吗','你还好吗','你',
             '你好不好','你好不好呢','我不开心']
    sentence = '你好吗'
    results = get_similarity_topN_new(sentence,texts)
    for l in results:
        print(l)   
    
    # 两个句子之间
    sentence1 = '你好吗'
    sentence2 = '你过的好吗'
    print(get_similarity_two(sentence1,sentence2))

