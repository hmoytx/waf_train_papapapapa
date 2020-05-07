import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from CNN import *
from wordcev import *
from torchtext import data
from torchtext.vocab import Vectors
from data import *
from torch import nn
import torch
from torchtext.data import Iterator, BucketIterator
from torchtext import datasets
from torchtext.vocab import GloVe
import torchtext
import pandas as pd
import re
# from gensim.test.utils import datapath, get_tmpfile
# from gensim.models import KeyedVectors
# # 已有的glove词向量
# glove_file = datapath('glove.6B.100d.txt')
# # 指定转化为word2vec格式后文件的位置
# tmp_file = get_tmpfile("D:/pytorch_test/attack_trainword2vec.txt")
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_file, tmp_file)



device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def eval(data_iter, model):
    model.eval()
    corrects, avg_loss, Tp, Fn, Tn, Fp= 0, 0, 0, 0, 0, 0
    for batch in data_iter:
        feature, target = batch.content, batch.type
        feature.data.t_()
        feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target)
        avg_loss += loss.item()
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
        Tp  += ((result.view(target.size()).data == 1) & (target.data == 1)).sum()
        Fn  += ((result.view(target.size()).data == 0) & (target.data==1)).sum()
        Tn += ((result.view(target.size()).data == 0) & (target.data == 0)).sum()
        Fp += ((result.view(target.size()).data == 1) & (target.data == 0)).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    # p = Tp / (Tp + Fp)
    # r = Tp / (Tp + Fn)
    # F1 = 2 * r * p / (r + p)

    print('\nEvaluation - loss: {:.6f} acc: {:.4f}%({}/{} {} {} {}) \n'.format(avg_loss, accuracy, corrects, size, Tp, Fp, Fn, Tn))
    return accuracy,Tp, Fp, Fn, Tn


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir,save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix,steps)
    torch.save(model.state_dict(),save_path)


def train(train_iter, dev_iter, epochs, model, log_interval, dev_interval, save_best, early_stop, save_interval, save_dir="./result", device=device_gpu):

    model.cuda(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    print('training...')
    for epoch in range(1, epochs + 1):
        for batch in train_iter:
            feature, target = batch.content,batch.type  # (W,N) (N)
            feature.data.t_()  # 转置使得维度匹配


            feature, target = feature.cuda(), target.cuda()

            logit = model(feature)  ##输出结果
            loss = F.cross_entropy(logit, target)  ## 计算loss
            loss.backward()  ##计算梯度
            optimizer.step()  ## 更新参数

            steps += 1
            if steps % log_interval == 0:  ##输出当前训练集上的效果
                result = torch.max(logit, 1)[1].view(target.size())
                corrects = (result.data == target.data).sum()
                accuracy = corrects * 100.0 / batch.batch_size
                sys.stdout.write('\rBatch[{}] - loss: {:.6f} acc: {:.4f}$({}/{})'.format(steps,
                                                                                         loss.data[0].item(),
                                                                                         accuracy,
                                                                                         corrects,
                                                                                         batch.batch_size))
            if steps % dev_interval == 0:
                dev_acc = eval(dev_iter, model)  ##得到当前验证集上的效果
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if save_best:
                        save(model, save_dir, 'best', steps)
                else:
                    if steps - last_step >= early_stop:
                        print('early stop by {} steps.'.format(early_stop))
            elif steps % save_interval == 0:  ##保存模型
                save(model, save_dir, 'snapshot', steps)




if __name__ == "__main__":

    tokenize = lambda x: x.split()
    device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # vectors = Vectors(name='D:/pytorch_test/attack_trainword2vec.txt')
    TEXT = data.Field(sequential=True, lower=True, fix_length=50)  # fix_length指定了每条文本的长度，截断补长
    LABEL = data.Field(sequential=False, use_vocab=False)

    train_data = pd.read_csv('./data3/train_xss.csv', encoding='ISO-8859-1')
    vaild_data = pd.read_csv('./data3/vaild_xss.csv', encoding='ISO-8859-1')
    test_data = pd.read_csv('./data3/test_xss.csv', encoding='ISO-8859-1')
    print(test_data.head(2))
    train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
    vaild_examples, vaild_fields = get_dataset(vaild_data, TEXT, LABEL)
    test_examples, test_fields = get_dataset(test_data, TEXT, None, test=True)

    train = data.Dataset(train_examples, train_fields)
    vaild = data.Dataset(vaild_examples, vaild_fields)
    test = data.Dataset(test_examples, test_fields)

    TEXT.build_vocab(train, vectors="glove.6B.200d")


    for i in range(0,len(train)):
        train[i].content = tuple(train[i].content)
        # train[i].type = int(train[i].type)
    for i in range(0,len(vaild)):
        vaild[i].content = tuple(vaild[i].content)
        # vaild[i].type = int(vaild[i].type)
    for i in range(0,len(test)):
        test[i].content = tuple(test[i].content)
        # test[i].type = int(test[i].type)
    train_iter, val_iter = BucketIterator.splits(
        (train, vaild),  # 构建数据集所需的数据集
        batch_sizes=(50, 50),
        device=1,  # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.content),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
    )
    test_iter = Iterator(test, batch_size=50, device=1, sort=False, sort_within_batch=False, repeat=False)


    # weight_matrix = TEXT.vocab.vectors
    # print(train.__dict__.keys())
    # print(train[5].label, train[5].payload)
    # TEXT = data.Field(sequential=True)
    # 以下两种指定预训练词向量的方式等效
    # TEXT.build_vocab(train, vectors="glove.6B.200d")


    weight_matrix = TEXT.vocab.vectors


    print(train[1111].type)
    model = textCNN(1, len(TEXT.vocab), [2,2,3,4,5], 16, 0.5)
    # train(train_iter, val_iter, 100, model, 100, 10, 1, 0, 10)

    model.cuda(device_gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    global avg_corr, a_Tp, a_Fp, a_Fn, a_Tn
    avg_corr= 0
    a_Tp, a_Fp, a_Fn, a_Tn = 0, 0, 0, 0
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    print('training...')
    for epoch in range(1, 20 + 1):
        for batch in train_iter:
            feature, target = batch.content,batch.type  # (W,N) (N)
            feature.data.t_()  # 转置使得维度匹配
            feature, target = feature.cuda(), target.cuda()
            logit = model(feature)  ##输出结果
            loss = F.cross_entropy(logit, target)  ## 计算loss
            loss.backward()  ##计算梯度
            optimizer.step()  ## 更新参数
            steps += 1

            if steps == 1:
                print(feature)
            if steps % 100 == 0:  ##输出当前训练集上的效果
                result = torch.max(logit, 1)[1].view(target.size())
                corrects = (result.data == target.data).sum()
                accuracy = corrects * 100.0 / batch.batch_size
                sys.stdout.write('\rBatch[{}] - loss: {:.6f} acc: {:.4f}%({}/{})'.format(steps,
                                                                                         loss.item(),
                                                                                         accuracy,
                                                                                         corrects,
                                                                                         batch.batch_size))
            if steps % 200 == 0:

                dev_acc, Tp, Fp, Fn, Tn= eval(val_iter, model)  ##得到当前验证集上的效果
                a_Tp += Tp
                a_Fp += Fp
                a_Fn += Fn
                a_Tn += Tn
                print(a_Tp, a_Fp, a_Fn, a_Tn)
            #     if dev_acc > best_acc:
            #         best_acc = dev_acc
            #         last_step = steps
            #
            #         save(model, './result', 'best', steps)
            #     else:
            #         if steps - last_step >= 1000:
            #             print('early stop by {} steps.'.format(1))
            # elif steps % 2000 == 0:  ##保存模型
    save(model, './result', 'snapshot', steps)

