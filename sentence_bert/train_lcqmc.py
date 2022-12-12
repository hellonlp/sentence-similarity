# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:32:43 2022

@author: chenming
"""

# LCQMC微调预训练模型

import sys
import numpy as np
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from keras.layers import Input, Dense, Lambda
from keras.initializers import TruncatedNormal
from sentence_bert.modules import get_tokenizer, convert_to_ids, get_encoder
from sentence_bert.modules import Model, merge
from sentence_bert.utils import load_lcqmc_train_data
from sentence_bert.hyperparameters import Hyperparamters as hp




# 加载LQCMC
data_path = 'datasets/chn/senteval_cn/'
train_data_0 = load_lcqmc_train_data(data_path + 'LCQMC/LCQMC.train.data')
test_data_0 = load_lcqmc_train_data(data_path + 'LCQMC/LCQMC.test.data')
valid_data_0 = load_lcqmc_train_data(data_path + 'LCQMC/LCQMC.valid.data')
train_data = train_data_0 + test_data_0 + valid_data_0


# 建立分词器
tokenizer = get_tokenizer(hp.dict_path)

# 数据转换为ID
a_token_ids, b_token_ids, labels = convert_to_ids(train_data, tokenizer)
label2id = {0.0: 0, 1.0: 1}
labels = np.array([[label2id[l]] for l in labels])



# 建立模型
encoder = get_encoder(hp.config_path, hp.checkpoint_path, 'last-avg')

t1_in = Input(shape=(None,))
t2_in = Input(shape=(None,))
s1_in = Input(shape=(None,))
s2_in = Input(shape=(None,))

z1 = encoder([t1_in, s1_in])
z2 = encoder([t2_in, s2_in])
z = Lambda(merge)([z1, z2])
p = Dense(
    units=3,
    activation='softmax',
    use_bias=False,
    kernel_initializer=TruncatedNormal(stddev=0.02)
)(z)

train_model = Model([t1_in, t2_in, s1_in, s2_in], p)
train_model.summary()

AdamW = extend_with_weight_decay(Adam, name='AdamW')
AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
train_steps = int(len(train_data) / hp.batch_size * hp.epochs)
warmup_steps = int(train_steps * hp.warmup_proportion)
optimizer = AdamWLR(
    learning_rate=hp.learning_rate,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    lr_schedule={warmup_steps: 1, train_steps: 0}
)
train_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

a_segment_ids = np.zeros_like(a_token_ids)
b_segment_ids = np.zeros_like(b_token_ids)

# 训练模型
train_model.fit([a_token_ids, b_token_ids, a_segment_ids, b_segment_ids],
                labels,
                epochs=hp.epochs,
                batch_size=hp.batch_size)

# 保存模型
encoder.save_weights(hp.model_save_path)



