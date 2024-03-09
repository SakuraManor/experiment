import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import re
import operator
import os, sys
import math
from xml.dom.minidom import parse
import xml.dom.minidom

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Import our models. The package will take care of downloading the models automaticallycd
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large", local_files_only=True)
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large", local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
# model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

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
            text = text.replace('*', '')
            text = text.replace("(", "")
            text = text.replace(")", "")
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
            text = text.replace('*', "")
            for word in line.split():
                word = word.strip()
                # if word in stopwords:
                #     continue
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


# Tokenize input texts
source_inputs = tokenizer(sources, padding=True, truncation=True, return_tensors="pt")
target_inputs = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    source_embeddings = model(**source_inputs, output_hidden_states=True, return_dict=True).pooler_output
    target_embeddings = model(**target_inputs, output_hidden_states=True, return_dict=True).pooler_output


k1set = [0.05, 0.10, 0.15]
k2set = [0.01, 0.03, 0.05]
for k1 in k1set:
    for k2 in k2set:
        print("========================================================================================================")
        print("k1: " + str(k1) + "," + "k2: " + str(k2))
        all = 0
        idf_map = {}
        dict_idf = {}
        for i in range(0, len(targets)):
            sum = 0
            i_dict = {}
            for j in range(0, len(targets)):
                cosine_sim = 1 - cosine(target_embeddings[i],  target_embeddings[j])
                i_dict[targetid[j]] = cosine_sim
            sim_list = sorted(i_dict.items(), key=operator.itemgetter(1), reverse=True)

            for j in range(0, math.ceil(len(targets)*k1)):
                if not idf_map.__contains__(sim_list[j][0]):
                    idf_map[sim_list[j][0]] = 1
                else:
                    idf_map[sim_list[j][0]] = idf_map.get(sim_list[j][0]) + 1

        for key in idf_map.keys():
            dict_idf[key] = math.log(len(targets)/(idf_map.get(key)))


        kkk = 0
        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        ap = 0
        result = {0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0, 1: 0, "pre1": 0}
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
                    cosine_sim_0_1 = 1 - cosine(source_embeddings[i], target_embeddings[j])
                    dict[targetid[j]] = cosine_sim_0_1
                list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

                # 奖励策略
                dont_sort = []
                sum = 0
                for o in range(0, round(len(targets) * k2)):
                    if o == 0:
                        top = list[o][1]
                    label = list[o][0]
                    dont_sort.append(label)
                new_dict = {}
                key = 0
                for o in range(0, round(len(targets) * k2)):
                    label = list[o][0]
                    index = label_text.get(label)
                    for p in range(0, len(targets)):
                        if index == p:
                            continue
                        cosine_sim = 1 - cosine(target_embeddings[index],  target_embeddings[p])
                        new_dict[targetid[p]] = cosine_sim

                    new_list = sorted(new_dict.items(), key=operator.itemgetter(1), reverse=True)

                    sum_new = 0
                    for q in range(0, round(len(new_list) * k1)):
                        sum_new = sum_new + dict_idf.get(new_list[q][0])

                    for q in range(0, round(len(new_list) * k1)):
                        name = new_list[q][0]
                        if dont_sort.__contains__(name) or dict.get(label) < dict.get(name):
                            continue
                        k = dict_idf.get(name) / sum_new

                        dict[name] = dict.get(name) + ((top - dict.get(name)) * k)

                list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
                curCorrect = 0
                dict = {}
                n = 0
                level = 0
                allPrecision = 0
                for j in range(0, len(list)):
                    k = j
                    level = j
                    if k != len(list) - 1:
                        while list[k][1] == list[k + 1][1]:
                            if answerSet.__contains__(list[k][0]):
                                tmp = list[j]
                                list[j] = list[k]
                                list[k] = tmp
                                break
                            k += 1
                    if answerSet.__contains__(list[j][0]):
                        curCorrect = curCorrect + 1
                        curPrecision = curCorrect / (j + 1)
                        allPrecision += curPrecision

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
                    if 1 == curCorrect and not dict.__contains__("pre1"):
                        dict["pre1"] = curCorrect / (level + 1)


                result[0.2] = (result.get(0.2) + dict.get(0.2))
                result[0.4] = (result.get(0.4) + dict.get(0.4))
                result[0.6] = (result.get(0.6) + dict.get(0.6))
                result[0.8] = (result.get(0.8) + dict.get(0.8))
                result[1] = (result.get(1) + dict.get(1))
                result["pre1"] = (result.get("pre1") + dict.get("pre1"))

                ap = ap + (allPrecision / len(answerSet))

        map = ap / kkk
        print("map: %.3f" % map)

        for key, value in result.items():
            print(f"{key}: " + "%.3f" % (value/kkk))
