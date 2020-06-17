# NewEmbeddings

博客地址：https://blog.csdn.net/weixin_41089007/article/details/106604465

## 代码介绍

dataProcess.py    处理原始数据，生成训练集测试集和训练word2vec模型的txt

generatePretrainedMatrix.py   生成word2vec.bin, attn_embedding.npy

generateEmbedding.py    生成word2idx, label2idx, word_embedding.npy, tfcr_embedding.npy

train.py    主函数

## 运行步骤
dataProcess.py >> generatePretrainedMatrix.py >> generateEmbedding.py >> train.py
