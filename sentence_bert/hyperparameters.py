# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:49:38 2022

@author: chenming
"""



import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)


class Hyperparamters:
    # RoBERTa parameters
    model_type = 'base'
    assert model_type in ['base', 'large']
    if model_type == 'base':
        model_name = 'roberta_base_zh'
    elif model_type == 'large':
        model_name = 'roberta_large_zh'
    config_path = os.path.join(pwd, '%s/bert_config.json' % model_name)
    checkpoint_path = os.path.join(pwd, '%s/bert_model.ckpt' % model_name)
    dict_path = os.path.join(pwd, '%s/vocab_chinese.txt' % model_name) 
    
    # Train parameters
    epochs = 3
    learning_rate = 2e-5
    batch_size = 16
    warmup_proportion = 0.1
    maxlen = 64
    k = 384
    
    # Weights path
    use_weigths = True
    weights_save_path = os.path.join(pwd, 'weights',model_name + '_lcqmc.weights')
    weights_load_path = os.path.join(pwd, 'weights', 'roberta_base_zh_lcqmc.weights')
    
    # Model path
    model_load_path = os.path.join(pwd, 'model_bert_whitening', 'roberta', 'V3.0', 'model_roberta_whitening.npz')
    vectors_path = os.path.join(pwd, 'model_bert_whitening', 'roberta', 'V3.0', 'vector_roberta_whitening.npz')

    ## Model path
    #model_load_path = os.path.join(pwd, 'model_bert_whitening', 'simbert', 'V2.0', 'model_roberta_whitening.npz')
    #vectors_path = os.path.join(pwd, 'model_bert_whitening', 'simbert', 'V2.0', 'vector_roberta_whitening.npz')


if __name__ == '__main__': 
    #
    hp = Hyperparamters()
    print(hp.config_path)
    print(hp.weights_load_path)
    
    
