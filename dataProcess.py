import pandas as pd
from bs4 import BeautifulSoup


def cleanReview(subject):
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',',
                                                                                                          '').replace(
        '.', '').replace('?', '').replace('(', '').replace(')', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)

    return newSubject


data = pd.read_csv('data/IMDBDataset.csv')
data["review"] = data["review"].apply(cleanReview)
# 将有标签的数据和无标签的数据合并
df = data['review']
# 保存成txt文件
df.to_csv("data/wordEmbedding.txt", index=False)

import pandas as pd
from bs4 import BeautifulSoup


def cleanReview(subject):
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',',
                                                                                                          '').replace(
        '.', '').replace('?', '').replace('(', '').replace(')', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)

    return newSubject


data = pd.read_csv('data/IMDBDataset.csv')
data["review"] = data["review"].apply(cleanReview)
train, test = data[:int(0.8 * data.shape[0])], data[int(0.8 * data.shape[0]):]
print(train.shape, test.shape)
train.to_csv('data/train_data.csv', index=False)
test.to_csv('data/test_data.csv', index=False)
