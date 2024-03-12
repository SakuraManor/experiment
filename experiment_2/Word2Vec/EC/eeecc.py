import torch
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import re
import operator
import os, sys
import math
from xml.dom.minidom import parse
import xml.dom.minidom

from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim

# Using the pre-trained word2vec model trained using Google news corpus of 3 billion running words.
# The model can be downloaded here: https://bit.ly/w2vgdrive (~1.4GB)
# Feel free to use to your own model.
googlenews_model_path = '../data/GoogleNews-vectors-negative300.bin'
stopwords_path = "../data/stopwords_en.txt"

model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)


COOKED_FOLDER = './answerFile/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

answer = {}

for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    with open(filepath, 'r', encoding='utf-8') as f:  # 读取文件
        data = f.readlines()
        f.close()  # 关
        # 将文件转换成字符串
        for line in data:
            line = line.replace(":", " ")
            i = 0
            key = ""
            for word in line.split(" "):
                if i == 0:
                    key = word
                    answer[key] = []
                elif i == len(line.split(" ")) - 1:
                    continue
                else:
                    answer.get(key).append(word)
                i += 1
# 打开文件
COOKED_FOLDER = './source/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)
sourceid = []
sources = []

for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    with open(filepath, 'r', encoding='utf-8') as f:  # 读取文
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
        sourceid.append(file)
        sources.append(text)

# 打开文件
COOKED_FOLDER = './target/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)
targetid = []
targets = []

for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    with open(filepath, 'r', encoding='utf-8') as f:  # 读取文
        data = f.readlines()
        f.close()  # 关
        # 将文件转换成字符串
        text = ""
        i = 0
        for line in data:
            newline = ""
            line = line.strip()
            line = line.replace('`', "")
            line = line.replace('(', "")
            line = line.replace(')', "")
            for word in line.split():
                word = word.strip()
                if word.isdigit() or word.__contains__('/') or word == 'Version' \
                    or word == 'PIN':
                    continue
                elif word[0].isupper() and len(word) == 1:
                    newline += word
                else:
                    newline += word + " "
            text += newline
        targetid.append(file)
        targets.append(text)


source_embeddings = model.encode(sources)
target_embeddings = model.encode(targets)

all = 0
idf_map = {}
dict_idf = {}
for i in range(0, len(targets)):
    sum = 0
    i_dict = {}
    for j in range(0, len(targets)):
        cosine_sim = ds.calculate_similarity(targets[i],  targets[j])
        i_dict[targetid[j]] = cosine_sim
    sim_list = sorted(i_dict.items(), key=operator.itemgetter(1), reverse=True)

    all += math.ceil(len(targets)*0.1)
    for j in range(0, math.ceil(len(targets)*0.1)):
        if not idf_map.__contains__(sim_list[j][0]):
            idf_map[sim_list[j][0]] = 1
        else:
            idf_map[sim_list[j][0]] = idf_map.get(sim_list[j][0]) + 1

for key in idf_map.keys():
    dict_idf[key] = math.log((idf_map.get(key))/(len(targets)+1))

kkk = 0
# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
result = {0.1: 0, 0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0, 1: 0}
for i in range(0, len(sources)):
    flag = False
    if answer.__contains__(sourceid[i]) and len(answer.get(sourceid[i])) > 1:
        kkk += 1
        answerSet = []
        for ans in answer.get(sourceid[i]):
            answerSet.append(ans)

        dict = {}
        label_text = {}
        top = 0
        for j in range(0, len(targets)):
            label_text[targetid[j]] = j
            cosine_sim_0_1 = ds.calculate_similarity(sources[i], targets[j])
            dict[targetid[j]] = cosine_sim_0_1
        list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

        # if sourceid[i] == "20.txt":
        #     print(list)

        # dont_sort = []
        # sum = 0
        # for o in range(0, math.ceil(len(targets) * 0.01)):
        #     if o == 0:
        #         top = list[o][1]
        #     label = list[o][0]
        #     dont_sort.append(label)
        #     sum = sum + dict_idf.get(label)
        #
        # new_dict = {}
        # key = 0
        # for o in range(0, math.ceil(len(targets) * 0.01)):
        #     label = list[o][0]
        #     k = (dict_idf.get(label) / sum)
        #     index = label_text.get(label)
        #     for p in range(0, len(targets)):
        #         if index == p:
        #             continue
        #         cosine_sim = 1 - cosine(target_embeddings[index],  target_embeddings[p])
        #         new_dict[targetid[p]] = cosine_sim
        #
        #     new_list = sorted(new_dict.items(), key=operator.itemgetter(1), reverse=True)
        #
        #     for q in range(0, math.ceil(len(new_list) * 0.05)):
        #         name = new_list[q][0]
        #         if dont_sort.__contains__(name) or dict.get(label) < dict.get(name):
        #             continue
        #         dict[name] = dict.get(name) + ((dict.get(label) - dict.get(name)) * k)
        #         if k == 0:
        #             flag = True
        #
        #     list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

        curCorrect = 0
        dict = {}
        n = 0
        for j in range(0, len(list)):
            level = j
            if j != len(list) - 1:
                while list[j][1] == list[j + 1][1]:
                    if answerSet.__contains__(list[j][0]):
                        break
                    j += 1
            if answerSet.__contains__(list[j][0]):
                curCorrect = curCorrect + 1
            if round(len(answerSet) * 0.1) == curCorrect and not dict.__contains__(0.1):
                dict[0.1] = curCorrect / (level + 1)
            if round(len(answerSet) * 0.2) == curCorrect and not dict.__contains__(0.2):
                dict[0.2] = curCorrect / (level + 1)
            if round(len(answerSet) * 0.4) == curCorrect and not dict.__contains__(0.4):
                dict[0.4] = curCorrect / (level + 1)
            if round(len(answerSet) * 0.6) == curCorrect and not dict.__contains__(0.6):
                dict[0.6] = curCorrect / (level + 1)
            if round(len(answerSet) * 0.8) == curCorrect and not dict.__contains__(0.8):
                dict[0.8] = curCorrect / (level + 1)
            if round(len(answerSet) * 1) == curCorrect and not dict.__contains__(1):
                dict[1] = curCorrect / (level + 1)

        print(dict)
        result[0.1] = (result.get(0.1) + dict.get(0.1))
        result[0.2] = (result.get(0.2) + dict.get(0.2))
        result[0.4] = (result.get(0.4) + dict.get(0.4))
        result[0.6] = (result.get(0.6) + dict.get(0.6))
        result[0.8] = (result.get(0.8) + dict.get(0.8))
        result[1] = (result.get(1) + dict.get(1))


for value in result.values():
    print("%.3f" % (value / kkk))