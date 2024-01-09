# PromCSE(sup)

## Uses
To use the tool, first install the `promcse` package from [PyPI](https://pypi.org/project/promcse/)
```bash
pip install promcse
```

After installing the package, you can load our model by two lines of code
```python
from promcse import PromCSE
model = PromCSE("hellonlp/promcse-roberta-base-zh", "cls", 10)
```

Then you can use our model for encoding sentences into embeddings
```python
embeddings = model.encode("武汉是一个美丽的城市。")
print(embeddings.shape)
#torch.Size([768])
```

Compute the cosine similarities between two groups of sentences
```python
sentences_a = ['你好吗']
sentences_b = ['你怎么样','我吃了一个苹果','你过的好吗','你还好吗','你',
               '你好不好','你好不好呢','我不开心','我好开心啊', '你吃饭了吗',
               '你好吗','你现在好吗','你好个鬼']
similarities = model.similarity(sentences_a, sentences_b)
print(similarities)
#[[0.7818036 , 0.0754933 , 0.751326  , 0.83766925, 0.6286671 ,
#  0.917025  , 0.8861941 , 0.20904644, 0.41348672, 0.5587336 ,
#  1.0000001 , 0.7798723 , 0.70388055]]
