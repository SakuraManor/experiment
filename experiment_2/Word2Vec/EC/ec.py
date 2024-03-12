import torch
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import re
import operator
import os, sys
import math
from xml.dom.minidom import parse
import xml.dom.minidom

model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
# 打开文件
COOKED_FOLDER = './CodeDescription/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

text1s = []
label1s = []
# 输出所有文件和文件夹
for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    with open(filepath, 'r', encoding='utf-8') as f:  # 读取文件
        data = f.readlines()
        f.close()  # 关
        # 将文件转换成字符串
        text = ""
        for line in data:
            newline = ""
            line = line.strip()
            line = line.replace('`', "")
            for word in line.split():
                word = word.strip()
                if word.isdigit() or word.__contains__('/'):
                    continue
                elif word[0].isupper() and len(word) == 1:
                    newline += word
                else:
                    newline += word + " "
            text += newline
    text1s.append(text)
    label1s.append(file)


embeddings = model.encode(text1s)

dict1 = {}
all = 0
for i in range(0, len(text1s)):
    sum = 0
    for j in range(0, len(text1s)):
        if j == i:
            continue
        cosine_sim = 1 - cosine(embeddings[i], embeddings[j])
        sum = sum + cosine_sim
    dict1[label1s[i]] = (sum / (len(text1s) - 1))
    all = all + (sum / (len(text1s) - 1))

for key in dict1.keys():
    dict1[key] = dict1.get(key)/all

# for key in dict.keys():
#     print(key)
# for value in dict.values():
#     print(value)

idf = sorted(dict1.items(), key=operator.itemgetter(1), reverse=False)

dict_idf = {}

for j in range(0, len(idf)):
    len1 = len(text1s)
    tmp = idf[j][1] * math.log((len1 + 1)/(j+1))
    dict_idf[idf[j][0]] = tmp


# 打开文件
COOKED_FOLDER = './source/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

labels = []
texts = []
# 输出所有文件和文件夹
for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    with open(filepath, 'r', encoding='utf-8') as f:  # 读取文件
        data = f.readlines()
        f.close()  # 关
        # 将文件转换成字符串
        text = ""
        for line in data:
            newline = ""
            line = line.strip()
            line = line.replace('`', "")
            for word in line.split():
                word = word.strip()

                if word.isdigit() or word.__contains__('/'):
                    continue
                elif word[0].isupper() and len(word) == 1:
                    newline += word
                else:
                    newline += word + " "
            text += newline
    texts.append(text)
    labels.append(file)

# 打开文件
COOKED_FOLDER = './target/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

# 输出所有文件和文件夹
for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    with open(filepath, 'r', encoding='utf-8') as f:  # 读取文件
        data = f.readlines()
        f.close()  # 关
        # 将文件转换成字符串
        text = ""
        for line in data:
            newline = ""
            line = line.strip()
            line = line.replace('`', "")
            for word in line.split():
                word = word.strip()
                if word.isdigit() or word.__contains__('/'):
                    continue
                elif word[0].isupper() and len(word) == 1:
                    newline += word
                else:
                    newline += word + " "
            text += newline
    texts.append(text)
    labels.append(file)


embeddings = model.encode(texts)

label_text = {}
dict = {}
for i in range(1, len(texts)):
    label_text[labels[i]] = i
    cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[i])
    dict[labels[i]] = cosine_sim_0_1

list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)



sum = 0
for i in range(0, 5):
    label = list[i][0]
    sum = sum + dict_idf.get(label)

new_dict = {}
for i in range(0, 5):
    label = list[i][0]
    k = 1 + (dict_idf.get(label) / sum)
    index = label_text.get(label)
    for j in range(1, len(texts)):
        if index == j:
            continue
        cosine_sim = 1 - cosine(embeddings[index], embeddings[j])
        new_dict[labels[j]] = cosine_sim

    new_list = sorted(new_dict.items(), key=operator.itemgetter(1), reverse=True)
    for p in range(0, 10):
        label = new_list[p][0]
        dict[label] = k * ((dict.get(label) + new_list[p][1]) / 2)

list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

for i in range(0, len(list)):
    print(list[i][0])
for i in range(0, len(list)):
    print(list[i][1])

curCorrect = 0
corrects = ["66.txt", "67.txt", "68.txt", "69.txt", "70.txt", "71.txt", "72.txt",
            "73.txt", "74.txt", "75.txt", "76.txt", "77.txt", "78.txt", "79.txt"]
dict = {}
for j in range(0, len(list)):
    curLabel = list[j][0]
    if corrects.__contains__(curLabel):
        curCorrect = curCorrect + 1
    if math.ceil(len(corrects) * 0.1) == curCorrect and not dict.__contains__(0.1):
        dict[0.1] = curCorrect / (j + 1)
    if round(len(corrects) * 0.2) == curCorrect and not dict.__contains__(0.2):
        dict[0.2] = curCorrect / (j + 1)
    if math.ceil(len(corrects) * 0.4) == curCorrect and not dict.__contains__(0.4):
        dict[0.4] = curCorrect / (j + 1)
    if round(len(corrects) * 0.6) == curCorrect and not dict.__contains__(0.6):
        dict[0.6] = curCorrect / (j + 1)
    if round(len(corrects) * 0.8) == curCorrect and not dict.__contains__(0.8):
        dict[0.8] = curCorrect / (j + 1)
    if round(len(corrects) * 1) == curCorrect and not dict.__contains__(1):
        dict[1] = curCorrect / (j + 1)

for value in dict.values():
    print(value)
