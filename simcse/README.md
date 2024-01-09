# SimCSE(sup)


## Model List
The evaluation dataset is in Chinese.
|          Model          | STS-B(w-avg) | ATEC | BQ | LCQMC | PAWSX | Avg. |
|:-----------------------:|:------------:|:-----------:|:----------|:-------------|:------------:|:----------:|
|  BERT-Whitening  |  65.27| -| -| -| -| -|
|  SimBERT   |  70.01| -| -| -| -| -|
|  SBERT-Whitening  |  71.75| -| -| -| -| -|
|  [hellonlp/simcse-roberta-base-zh(sup)](https://huggingface.co/hellonlp/simcse-roberta-base-zh)  |  **80.96**| -| -| -| -| -|


## Uses
You can use our model for encoding sentences into embeddings
```python
import torch
from transformers import BertTokenizer
from transformers import BertModel
from sklearn.metrics.pairwise import cosine_similarity

# model
simcse_sup_path = "hellonlp/simcse-roberta-base-zh"
tokenizer = BertTokenizer.from_pretrained(simcse_sup_path)
MODEL = BertModel.from_pretrained(simcse_sup_path)

def get_vector_simcse(sentence):
    """
    预测simcse的语义向量。
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    output = MODEL(input_ids)
    return output.last_hidden_state[:, 0].squeeze(0)

embeddings = get_vector_simcse("武汉是一个美丽的城市。")
print(embeddings.shape)
#torch.Size([768])
```

You can also compute the cosine similarities between two sentences
```python
def get_similarity_two(sentence1, sentence2):
    vec1 = get_vector_simcse(sentence1).tolist()
    vec2 = get_vector_simcse(sentence2).tolist()
    similarity_list = cosine_similarity([vec1], [vec2]).tolist()[0][0]
    return similarity_list

sentence1 = '你好吗'
sentence2 = '你还好吗'
result = get_similarity_two(sentence1,sentence2)
print(result)
#0.848331
```

## 文章链接
- [论文解读]/(https://zhuanlan.zhihu.com/p/624425957)
- [论文复现]/(https://zhuanlan.zhihu.com/p/634871699)

## 模型链接
- [语言模型(simcse/model/loal_model)]/(https://pan.baidu.com/s/1bTqJBB49gwJksmUzNql5xg?pwd=siuj)
- [训练模型(simcse/roberta_pytorch)]/(https://pan.baidu.com/s/1NUMowTyAAa7sF-hHEcpUOg?pwd=eiqm)
- 
