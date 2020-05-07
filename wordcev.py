import torch
from torchtext import data
from torchtext.vocab import GloVe
from torchtext.data import Iterator
import numpy as np
from data import *
import urllib.parse

from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_dataset(csv_data, text_field, label_field, test=False):
	# id数据对训练在训练过程中没用，使用None指定其对应的field
    fields = [("id", None),
        ("content", text_field), ("type", label_field)]
    examples = []

    if test:
        # 如果为测试集，则不加载label
        for text in csv_data['content']:
            examples.append(data.Example.fromlist([None, clean_str(str(text)), None], fields))
    else:
        for text, label in zip(csv_data['content'], csv_data['type']):
            examples.append(data.Example.fromlist([None, clean_str(str(text)), label], fields))
    return examples, fields

# TEXT = data.Field(sequential=False, lower=True, fix_length=500) # fix_length指定了每条文本的长度，截断补长
# LABEL = data.Field(sequential=False, use_vocab=False)
#
#
# train_data = read_data('./result_sql.txt')
#
# train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
# #test_examples, test_fields = get_dataset(train_data, TEXT, None, test=True)
# train = data.Dataset(train_examples, train_fields)
#
# train_iter = Iterator(train, batch_size=8, device=device, sort=False, sort_within_batch=False, repeat=False)
#
# TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
# weight_matrix = TEXT.vocab.vectors
#
#
#
# print(train[5].payload)
