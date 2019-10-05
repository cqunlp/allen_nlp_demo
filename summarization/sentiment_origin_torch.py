import torch
import torch.nn as nn
import parser
import os
import jieba
import gensim

# data reader
def data_reader(path):
    data, label = [], []
    for i in os.listdir(path):
        with open(path + '/' + i, 'r', encoding='gbk') as f:
            try:
                content = f.read().strip()
                score = i.split('_')[-1].replace('.txt','')
                data.append(content)
                label.append(score)
            except Exception as e:
                # print(i,e)
                pass
    return data,label

# word spliter


def word_split():
    pass
# stop word dict
def stop_words_dict(filepath):
    stopwords = []
    for i in os.listdir(filepath):
        temp = [line.strip() for line in open(os.path.join(filepath,i), 'r', encoding='utf-8').readlines()]
        stopwords.extend(temp)
    return stopwords

# remove
def remove_stop_words(sentence):
    stopwords = stop_words_dict('/home/ubuntu/PycharmProjects/stopwords') 
    outstr = []
    for word in sentence:
        if word not in stopwords:
            if word != '\t'and'\n':
                outstr.append(word)
                # outstr += " "
    return outstr

# load data

path = '/media/ubuntu/Backup/Downloads/ctrip/'

train_path = path + 'train'
test_path = path + 'test'

train_x, train_y = data_reader(train_path)
test_x, test_y = data_reader(test_path)

# split words


# build wordvec



if __name__ == "__main__":
    pass