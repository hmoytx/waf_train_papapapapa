from wordcev import *
from torchtext import data
from data import *
from torch import nn
import torch
from torchtext.data import Iterator, BucketIterator
from torchtext import datasets
from torchtext.vocab import GloVe
import torchtext
import pandas as pd



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




learning_rate = 0.001
tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, lower=True, fix_length=100) # fix_length指定了每条文本的长度，截断补长
LABEL = data.Field(sequential=False, use_vocab=False)


train_data = pd.read_csv('./data/train.csv')

train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
#test_examples, test_fields = get_dataset(train_data, TEXT, None, test=True)

train = data.Dataset(train_examples, train_fields)

TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100))

for i in range(0,len(train)):
    train[i].content = tuple(train[i].content)
# for i in range(0,len(vaild)):
#     vaild[i].content = tuple(vaild[i].content)
# for i in range(0,len(test)):
#     test[i].content = tuple(test[i].content)

train_iter = Iterator(train, batch_size=10, device=-1, sort=False, sort_within_batch=False, repeat=False)
# weight_matrix = TEXT.vocab.vectors
# print(train.__dict__.keys())
# print(train[5].label, train[5].payload)

# 以下两种指定预训练词向量的方式等效
# TEXT.build_vocab(train, vectors="glove.6B.200d")



# model = ResNet(Residual_Block, [2, 2, 2, 2]).to(device)
# print(model)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#
print(len(train[1].content))
#
for idx, batch in enumerate(train_iter):
    print(batch)
    content, type = batch.content,batch.type
    print(content.shape, type.shape)