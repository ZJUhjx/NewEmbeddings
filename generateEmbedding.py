from collections import Counter
import json
import gensim
import pandas as pd
import numpy as np


def readStopWord(stopWordPath):
    """
    读取停用词
    """
    with open(stopWordPath, "r") as f:
        stopWords = f.read()
        stopWordList = stopWords.splitlines()
        # 将停用词用列表的形式生成，之后查找停用词时会比较快
        stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    return stopWordDict


def readData(filePath):
    """
    从csv文件中读取数据集
    """
    data = pd.read_csv(filePath)
    reviews = data['review'].tolist()
    reviews = [review.strip().split() for review in reviews]
    labels = data['sentiment'].tolist()

    return reviews, labels


def genEmbedding(reviews, labels, stopWordDict, embeddingSize, mode):
    """word,tfcr,attention
    生成词向量和词汇-索引映射字典，可以用全数据集
    """
    allWords = [word for review in reviews for word in review]

    # 去掉停用词
    subWords = [word for word in allWords if word not in stopWordDict]

    wordCount = Counter(subWords)  # 统计词频
    sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

    # 去除低频词
    words = [item[0] for item in sortWordCount if item[1] >= 3]

    if mode == 'word':
        vocab, wordEmbedding = getWordEmbedding(words, embeddingSize, 'model/word2Vec.bin')
    else:
        vocab, wordEmbedding = getWordEmbedding(words, embeddingSize, 'model/word2Vec.bin')
        wordEmbedding = getTfcrEmbedding(vocab, wordEmbedding)

    np.save('embeddings/' + str(mode) + '_embedding.npy', wordEmbedding)

    word2idx = dict(zip(vocab, list(range(len(vocab)))))

    uniqueLabel = list(set(labels))
    label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))

    # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
    with open("data/word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f)

    with open("data/label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f)

    return wordEmbedding


def getWordEmbedding(words, embeddingSize, path):
    """
    按照我们的数据集中的单词取出预训练好的word2vec中的词向量
    """
    vocab = []
    wordEmbedding = []
    word2Vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    vocab.append('PAD')
    vocab.append('UNK')
    wordEmbedding.append(np.zeros(embeddingSize))
    wordEmbedding.append(np.ones(embeddingSize))
    for word in words:
        try:
            vector = word2Vec.wv[word]
            vocab.append(word)
            wordEmbedding.append(vector)
        except:
            print(word + 'not in word vector')

    return vocab, np.array(wordEmbedding)


def getTfcrEmbedding(vocab, embedding_matrix):
    with open("data/pos_weight.json", "r", encoding="utf-8") as f:
        pos_weight = json.load(f)
    with open("data/neg_weight.json", "r", encoding="utf-8") as f:
        neg_weight = json.load(f)

    pos_embedding = np.zeros((embedding_matrix.shape[0], embedding_matrix.shape[1]))
    for i, word in enumerate(vocab):
        if word in ['PAD', 'UNK']:
            pos_embedding[i] = embedding_matrix[i]
        else:
            pos_embedding[i] = embedding_matrix[i] * pos_weight.get(word, 0)  # 没有则代表0权重

    neg_embedding = np.zeros((embedding_matrix.shape[0], embedding_matrix.shape[1]))
    for i, word in enumerate(vocab):
        if word in ['PAD', 'UNK']:
            neg_embedding[i] = embedding_matrix[i]
        else:
            neg_embedding[i] = embedding_matrix[i] * neg_weight.get(word, 0)

    tfcr_embedding = np.hstack([pos_embedding, neg_embedding])

    return tfcr_embedding


def genWordWeight(dataSource, stopWordDict):
    data = pd.read_csv(dataSource)
    positive = data[data['sentiment'] == 'positive']
    negative = data[data['sentiment'] == 'negative']

    review_pos = positive['review'].tolist()
    review_pos = [review.strip().split() for review in review_pos]
    review_pos = [word for review in review_pos for word in review]
    review_pos = [word for word in review_pos if word not in stopWordDict]
    count_pos = Counter(review_pos)
    review_neg = negative['review'].tolist()
    review_neg = [review.strip().split() for review in review_neg]
    review_neg = [word for review in review_neg for word in review]
    review_neg = [word for word in review_neg if word not in stopWordDict]
    count_neg = Counter(review_neg)

    word_weight = {}
    for word in count_pos:
        wc = count_pos[word]
        nc = len(count_pos)
        w = count_pos[word] + count_neg[word]
        weight = wc * wc / (nc * w)
        word_weight[word] = weight
    with open("data/pos_weight.json", "w", encoding="utf-8") as f:
        json.dump(word_weight, f)

    word_weight = {}
    for word in count_neg:
        wc = count_neg[word]
        nc = len(count_neg)
        w = count_pos[word] + count_neg[word]
        weight = wc * wc / (nc * w)
        word_weight[word] = weight
    with open("data/neg_weight.json", "w", encoding="utf-8") as f:
        json.dump(word_weight, f)


dataSource = "data/train_data.csv"
stopWordSource = "data/english.txt"
embeddingSize = 256

# 初始化停用词
stopWords = readStopWord(stopWordSource)
# 初始化数据集
reviews, labels = readData(dataSource)
# 初始化tfcr的词权重
genWordWeight(dataSource, stopWords)
# 初始化词汇-索引映射表和词向量矩阵 mode = ['word','tfcr']
genEmbedding(reviews, labels, stopWords, embeddingSize, mode='tfcr')
