import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F


class textCNN(nn.Module):
    def __init__(self, i_chanel, embed_num, kernel_sizes, kernel_num, Dropout, class_num=2):
        super(textCNN, self).__init__()


        Vocab = embed_num     ## 已知词的数量
        Dim = 200              ##每个词向量长度
        Cla = class_num       ##类别数
        Ci = i_chanel         ##输入的channel数
        Knum = kernel_num     ## 每种卷积核的数量
        Ks = kernel_sizes     ## 卷积核list，形如[2,3,4]
        dropout = Dropout
        self.embed = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=16,
        #         kernel_size=2,
        #         stride=1,
        #         padding=2,
        #     ),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, 32, 5, 1, 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(16, 32, 5, 1, 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(Ks) * Knum, Cla)  ##全连接层

    def forward(self, x):
        x = self.embed(x)  # (N,W,D)

        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)

        x = torch.cat(x, 1)  # (N,Knum*len(Ks))

        x = self.dropout(x)
        logit = self.fc(x)
        return logit

