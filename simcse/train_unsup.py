# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:08:58 2023

@author: cm
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import random
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from simcse.hyperparameters import Hyperparameters as hp
from simcse.utils import load_data_new
from simcse.networks import SimCSEModelUnsup



# tokenizer = BertTokenizer.from_pretrained(hp.model_path, use_fast=True)
tokenizer = BertTokenizer.from_pretrained(hp.pretrained_model_path, use_fast=True)


# 加载数据集
datasets_sts = {
    'STS-B-train': load_data_new('datasets/chn/senteval_cn/STS-B/STS-B.train.data'),
    'STS-B-test': load_data_new('datasets/chn/senteval_cn/STS-B/STS-B.test.data'),
    'STS-B-valid': load_data_new('datasets/chn/senteval_cn/STS-B/STS-B.valid.data')
}
datasets_lqcmc = {
    'LCQMC-train': load_data_new('datasets/chn/senteval_cn/LCQMC/LCQMC.train.data'),
    'LCQMC-test': load_data_new('datasets/chn/senteval_cn/LCQMC/LCQMC.test.data'),
    'LCQMC-valid': load_data_new('datasets/chn/senteval_cn/LCQMC/LCQMC.valid.data')
}


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data: List):
        self.data = data
      
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        return tokenizer([text, text], max_length=hp.MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])
    

class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=hp.MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        da = self.data[index]        
        return self.text_2_id([da[0]]), self.text_2_id([da[1]]), int(da[2])
    


def simcse_unsup_loss(y_pred):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0]) # [batch_size*2]
    y_true = (y_true - y_true % 2 * 2) + 1

    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1) # [batch_size*2, batch_size*2]

    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0]) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss


def eval_spearmanr(model, dataloader) -> float:
    """模型评估函数 
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=hp.DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(hp.DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(hp.DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(hp.DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(hp.DEVICE)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(hp.DEVICE)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(hp.DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)            
            label_array = np.append(label_array, np.array(label))
    # corrcoef 
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练函数"""
    model.train()
    global best
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(hp.DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(hp.DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(hp.DEVICE)
        print('='*20)
        print('=== input_ids ===',input_ids.shape)
        print('=== input_ids ===',input_ids)
        print('='*20)
        print('=== attention_mask ===',attention_mask.shape)
        print('=== attention_mask ===',attention_mask)
        output = model(input_ids, attention_mask, token_type_ids)        
        loss = simcse_unsup_loss(output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:     
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval_spearmanr(model, dev_dl)
            model.train()
            if best < corrcoef:
                best = corrcoef
                torch.save(model.state_dict(), hp.SAVE_PATH)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
       


if __name__ == '__main__':
    # load data
    logger.info(f'Device: {hp.DEVICE}, Pooling: {hp.POOLING}, Model path: {hp.pretrained_model_path}')
    train_data_lcqmc = datasets_lqcmc['LCQMC-train']#[:1000]
    train_data_sts = datasets_sts['STS-B-train']#[:1000]
    train_data = [l[0] for l in train_data_lcqmc] + [l[0] for l in train_data_sts]   # 两个数据集组合
    print(len(train_data_lcqmc),len(train_data_sts),len(train_data))
    train_data = random.sample(train_data, hp.SAMPLES)   # 随机采样    
    print(len(train_data))
    dev_data = datasets_sts['STS-B-valid']
    test_data = datasets_sts['STS-B-test']
    #
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=hp.BATCH_SIZE)
    dev_dataloader = DataLoader(TestDataset(dev_data), batch_size=hp.BATCH_SIZE)
    test_dataloader = DataLoader(TestDataset(test_data), batch_size=hp.BATCH_SIZE)
    # load model
    assert hp.POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimCSEModelUnsup(pretrained_model=hp.pretrained_model_path, pooling=hp.POOLING).to(hp.DEVICE)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.LR)
    # train
    best = 0
    for epoch in range(hp.EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer)
    logger.info(f'Train is finished, best model is saved at {hp.SAVE_PATH}')
