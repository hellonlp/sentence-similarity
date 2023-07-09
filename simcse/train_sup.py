# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:57:26 2023

@author: Chen Ming
"""



import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import random
# import time
from typing import Dict, List
# import jsonlines
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
from simcse.utils import load_data_new, load_snli_data
from simcse.networks import SimCSEModelSup


tokenizer = BertTokenizer.from_pretrained(hp.pretrained_model_path, use_fast=True)



# 加载数据集
datasets_sts = {
    'STS-B-train': load_data_new('F:/celery/datasets/chn/senteval_cn/STS-B/STS-B.train.data'),
    'STS-B-test': load_data_new('F:/celery/datasets/chn/senteval_cn/STS-B/STS-B.test.data'),
    'STS-B-valid': load_data_new('F:/celery/datasets/chn/senteval_cn/STS-B/STS-B.valid.data')
}
datasets_lqcmc = {
    'LCQMC-train': load_data_new('F:/celery/datasets/chn/senteval_cn/LCQMC/LCQMC.train.data'),
    'LCQMC-test': load_data_new('F:/celery/datasets/chn/senteval_cn/LCQMC/LCQMC.test.data'),
    'LCQMC-valid': load_data_new('F:/celery/datasets/chn/senteval_cn/LCQMC/LCQMC.valid.data')
}
datasets_snli = {
    'SNLI-train': load_snli_data('F:/celery/datasets/chn/senteval_cn/SNLI/train.txt'),
    'SNLI-test': load_snli_data('F:/celery/datasets/chn/senteval_cn/SNLI/test.txt'),
    'SNLI-valid': load_snli_data('F:/celery/datasets/chn/senteval_cn/SNLI/dev.txt')
}



class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer([text[0], text[1], text[2]], max_length=hp.MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])
     
    
class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=hp.MAXLEN, truncation=True, 
                         padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id([line[0]]), self.text_2_id([line[1]]), int(line[2])
    

                  

  
    
def simcse_sup_loss(y_pred):
    """
    有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=hp.DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=hp.DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss
    

def eval_spearmanr(model, dataloader) -> float:
    """模型评估函数 
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=hp.DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(hp.DEVICE)
            source_attention_mask = source['attention_mask'].squeeze(1).to(hp.DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(hp.DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target['input_ids'].squeeze(1).to(hp.DEVICE)
            target_attention_mask = target['attention_mask'].squeeze(1).to(hp.DEVICE)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(hp.DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))  
    # corrcoef       
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
        

def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(hp.DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(hp.DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(hp.DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval_spearmanr(model, dev_dl)
            model.train()
            if best < corrcoef:
                early_stop_batch = 0
                best = corrcoef
                torch.save(model.state_dict(), hp.SAVE_PATH_SUP)# + str(batch_idx)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                continue
            early_stop_batch += 1
            if early_stop_batch == 10:
                logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                logger.info(f"train use sample number: {(batch_idx - 10) * hp.BATCH_SIZE}")
                return 
            
            
            


if __name__ == '__main__':
    
    
    logger.info(f'device: {hp.DEVICE}, pooling: {hp.POOLING}, model path: {hp.pretrained_model_path}')
    tokenizer = BertTokenizer.from_pretrained(hp.pretrained_model_path)
    # load data (train)
    train_data = datasets_snli['SNLI-train']
    # train_data = datasets_lqcmc['LCQMC-train']
    random.shuffle(train_data)     
    # load data (test)                   
    # dev_data = load_data('sts', STS_DEV)
    # test_data = load_data('sts', STS_TEST)    
    dev_data = datasets_sts['STS-B-valid']
    test_data = datasets_sts['STS-B-test']
    #
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=hp.BATCH_SIZE)
    dev_dataloader = DataLoader(TestDataset(dev_data), batch_size=hp.BATCH_SIZE)
    test_dataloader = DataLoader(TestDataset(test_data), batch_size=hp.BATCH_SIZE)
    # load model    
    assert hp.POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimCSEModelSup(pretrained_model=hp.pretrained_model_path, pooling=hp.POOLING)
    model.to(hp.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.LR)
    # train
    best = 0
    for epoch in range(hp.EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {hp.SAVE_PATH_SUP}')
    # eval
    model.load_state_dict(torch.load(hp.SAVE_PATH_SUP))
    dev_corrcoef = eval_spearmanr(model, dev_dataloader)
    test_corrcoef = eval_spearmanr(model, test_dataloader)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')
    

