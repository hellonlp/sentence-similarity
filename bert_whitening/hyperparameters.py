# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:59:52 2022

"""



import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)



class Hyperparamters:
    # # RoBERTa parameters
    # model = 'roberta_base_zh'#'albert_small_zh_google'
    # bert_path = os.path.join(pwd, model)
    # vocab_path = os.path.join(pwd, model, 'vocab_chinese.txt')
    # checkpoint_path = os.path.join(pwd, model, 'bert_model.ckpt')
    # config_path = os.path.join(pwd, model, 'bert_config.json')

    # SimBERT parameters
    model = 'chinese_simbert_L-12_H-768_A-12'
    bert_path = os.path.join(pwd, model)
    vocab_path = os.path.join(pwd, model, 'vocab.txt')
    checkpoint_path = os.path.join(pwd, model, 'bert_model.ckpt')
    config_path = os.path.join(pwd, model, 'bert_config.json')
    
    
    # BERT Whitening parameters
    maxlen = 64
    k = 384
    
    # Model path
    model_save_path = os.path.join(pwd, 'model', 'V1.0_b', 'model_STS-B1.npz')
    model_load_path = os.path.join(pwd, 'model', 'V2.0', 'model_V2.0.npz')
    vectors_path = os.path.join(pwd, 'model', 'V2.0', 'vector_V2.0.npz')


if __name__ == '__main__': 
    #
    hp = Hyperparamters()
    print(hp.config_path)

