# Sentence Similarity



## Data List
下面的数据集都是中文的。
|          Data          | size(train) | size(valid) | size(test) |
|:----------------------:|:----------:|:----------:|:----------:|
|   [ATEC](https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/1gmnyz9emqOXwaHhSM9CCUA%3Fpwd%3Db17c)   |  62477|  20000|  20000|
|   [BQ](https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/1M-e01yyy5NacVPrph9fbaQ%3Fpwd%3Dtis9)     | 100000|  10000|  10000|
|   [LCQMC](https://pan.baidu.com/s/16DfE7fHrCkk4e8a2j3SYUg?pwd=bc8w )                                      | 238766|   8802|  12500|
|   [PAWSX](https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/1ox0tJY3ZNbevHDeAqDBOPQ%3Fpwd%3Dmgjn)  |  49401|   2000|   2000|
|   [STS-B](https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/10yfKfTtcmLQ70-jzHIln1A%3Fpwd%3Dgf8y)  |   5231|   1458|   1361|
|   [SNLI](https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/1NOgA7JwWghiauwGAUvcm7w%3Fpwd%3Ds75v)   | 146828|   2699|   2618|
|   [MNLI](https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/1xjZKtWk3MAbJ6HX4pvXJ-A%3Fpwd%3D2kte)   | 122547|   2932|   2397|



## Model List
考虑到有些数据集的 test 集较小，可能会导致评估准确性偏差较大，所以这里的评估数据同时使用了train、valid和test，且最终评估结果采用了加权平均（w-avg）的方法得到。

### 基于RoBERTa Base 版本
这里使用相同的语言模型**RoBERTa Base**。
|          Model          |  ATEC | BQ| LCQMC | PAWSX| STS-B| Avg. |
|:-----------------------:|:------------:|:-----------:|:----------|:-------------|:------------:|:----------:|
|  BERT-Whitening  | -| -| -| -| 65.27| -|
|  SimBERT   | -| -| -| -| 70.01| -|
|  SBERT-Whitening  | -| -| -| -| 71.75| -|
|  [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh)  | -| -| -| -| 78.61| -|
|  [hellonlp/simcse-base-zh](https://huggingface.co/hellonlp/simcse-roberta-base-zh)  | -| -| -| -| 80.96| -|
|  [hellonlp/promcse-base-zh](https://huggingface.co/hellonlp/promcse-bert-base-zh)  | -| -| -| -| **81.57**| -|


### 基于RoBERTa Large 版本
这里使用相同的语言模型**RoBERTa Large**。
xx



