import logging
import gensim
from gensim.models import word2vec


# 使用gensim自带的包输出word2vec词向量模型
def generateWord2vec():
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
    sentences = word2vec.LineSentence("data/wordEmbedding.txt")
    # 训练模型，词向量的长度设置为256， 迭代次数为5，采用CBOW模型，模型保存为bin格式
    model = gensim.models.Word2Vec(sentences, size=256, min_count=3, window=5)
    model.wv.save_word2vec_format("model/word2Vec" + ".bin", binary=True)


generateWord2vec()

###########################################################################################################
# 实现attention word embedding
import torch
import torch.nn as nn  # 神经网络工具箱torch.nn
import torch.nn.functional as F  # 神经网络函数torch.nn.functional
import torch.utils.data as tud  # Pytorch读取训练集需要用到torch.utils.data类
import torch.nn.parameter as Parameter  # 参数更新和优化函数

from collections import Counter  # 统计词频
import json
import sklearn
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度函数

import pandas as pd
import numpy as np
import scipy  # 数据分析三件套

import random
import math  # 数学和随机离不开

USE_CUDA = torch.cuda.is_available()  # GPU可用的标志位
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if USE_CUDA:
    torch.cuda.manual_seed(1)

# 词向量相关的超参数
K = 5  # number of negative samples 负样本随机采样数量，文本量越大该参数设置越小
C = 5  # nearby words threshold 指定周围三个单词进行预测，类似于3gram，根据和博士同事交流，一遍使用奇数，如阿里云安全比赛中，CNN的Filter size使用的是3、5 、7 、9。
EMBEDDING_SIZE = 256  # 词向量维度
ATTN_SIZE = 50  # 注意力维度
MAX_VOCAB_SIZE = 58080  # the vocabulary size 词汇表多大

# 参数优化的超参数
NUM_EPOCHS = 5  # The number of epochs of training 所有数据的训练大轮次数，每一大轮会对所有的数据进行训练
BATCH_SIZE = 128  # the batch size 每一轮中的每一小轮训练128个样本
LEARNING_RATE = 0.2  # the initial learning rate #学习率

LOG_FILE = "word-embedding.log"

######################################################################################################################
# data process
with open('data/word2idx.json', "r", encoding="utf-8") as f:
    word_to_idx = json.load(f)
idx_to_word = dict(zip(word_to_idx.values(), word_to_idx.keys()))
vocab = word_to_idx.keys()

data = pd.read_csv('data/train_data.csv')
reviews = data['review'].tolist()
reviews = [review.strip().split() for review in reviews]
allWords = [word for review in reviews for word in review]

wordCount = Counter(allWords)  # 统计词频
word_counts = np.array([wordCount[word] for word in vocab], dtype=np.float32)
word_freqs = word_counts / sum(word_counts)

VOCAB_SIZE = len(idx_to_word)  # 获得词典的实际长度


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_counts, word_freqs):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
            感觉并不需要存idx_to_word，相关信息已经包含在word_to_idx中了
        '''
        super().__init__()  # 通过父类初始化模型，然后重写两个方法
        self.text_encoded = [word_to_idx.get(word, word_to_idx['UNK']) for word in text]  # 把词数字化表示
        self.text_encoder = torch.Tensor(self.text_encoded).long()  # 转变为LongTensor，为什么要这样转换，是为了增大存储单词量

        self.word_to_idx = word_to_idx  # 保存数据
        self.idx_to_word = idx_to_word  # 保存数据
        self.word_freqs = torch.Tensor(word_freqs)  # 保存数据
        self.word_counts = torch.Tensor(word_counts)  # 保存数据

    def __len__(self):
        # 魔法函数__len__
        return len(self.text_encoded)  # 所有单词的总数

    def __getitem__(self, idx):
        # 魔法函数__getitem__，这个函数跟普通函数不一样
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''

        center_word = self.text_encoded[idx]  # 取得中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # 注意左闭右开
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 不知是否有更好的方式

        pos_words = torch.tensor([self.text_encoded[i] for i in pos_indices])
        # 周围词索引，就是希望出现的正例单词
        # print(pos_words)

        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)  # 该步是基于paper中内容
        # 负例采样单词索引，torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标。
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大。
        # 每个正确的单词采样K个，pos_words.shape[0]是正确单词数量

        return center_word, pos_words, neg_words


dataset = WordEmbeddingDataset(allWords, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


######################################################################################################################
# define  attention word embedding model
class AttentionEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size, attn_size):
        ''' 初始化输入和输出embedding
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.attn_size = attn_size

        # 共享参数，in out embedding共用一个
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.k = nn.Embedding(self.vocab_size, self.attn_size)
        self.q = nn.Embedding(self.vocab_size, self.attn_size)

        initrange = 0.5 / self.embed_size
        self.in_embed.weight.data.uniform_(-initrange, initrange)

        initrange2 = 0.5 / self.attn_size
        self.k.weight.data.uniform_(-initrange2, initrange2)
        self.q.weight.data.uniform_(-initrange2, initrange2)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        b * 2c —— b * 2c * embed —— attn = b * 2c  n*d1 b*2c*d1
        return: loss
        '''

        input_embedding = self.in_embed(input_labels)  # B * embed_size
        pos_embedding = self.in_embed(pos_labels)  # B * (2*C) * embed_size
        neg_embedding = self.in_embed(neg_labels)  # B * (2*C * K) * embed_size

        input_k = self.k(input_labels)  # B * attn_size
        pos_k = self.q(pos_labels)  # B * (2*C) * attn_size

        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze(2)  # B * (2*C)
        attn_pos = torch.bmm(pos_k, input_k.unsqueeze(2)).squeeze(2)  # B * (2*C) 每个单词都有对应的权重
        log_pos = torch.mul(log_pos, attn_pos)  # 点乘
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze(2)  # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg

        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


model = AttentionEmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE, ATTN_SIZE)
if USE_CUDA:
    model = model.cuda()


######################################################################################################################
# define evaluate func
def find_nearest(word, embedding_weights):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]


######################################################################################################################
# training process
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    print(len(dataloader))
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with open(LOG_FILE, "a") as fout:
                fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
                print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))
        if i == 10000:
            break

embedding_weights = model.input_embeddings()
np.save("embeddings/attn_embedding.npy".format(EMBEDDING_SIZE), embedding_weights)
# torch.save(model.state_dict(), "data/attn_embedding.th".format(EMBEDDING_SIZE))

# evaluate trained model
for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
    print(word, find_nearest(word, embedding_weights))
