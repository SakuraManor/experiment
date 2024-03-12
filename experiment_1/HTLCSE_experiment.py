from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import re
import operator
import os, sys
import math
import torch


stopwords = []
# stopwords_path = "./data/stopwords_en.txt"
# with open(stopwords_path, 'r') as fh:
#     stopwords = fh.read().split(",")

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large", local_files_only=True)
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large", local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
# model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

# 打开文件
COOKED_FOLDER = './GanntDataset/high/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

high = []
highName = []
# 输出所有文件和文件夹
for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    highName.append(file)
    with open(filepath, 'r') as f:  # 读取文件
        data = f.readlines()
        f.close()  # 关
        # 将文件转换成字符串
        text = ""
        for line in data:
            line = line.replace("/", " or ")
            line = line.replace("(", "")
            line = line.replace(")", "")
            for word in line.split(" "):
                if word not in stopwords:
                    text += word + " "
        high.append(text)


# 打开文件
COOKED_FOLDER = './GanntDataset/low/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

low = []
lowName = []
answer = {}
# 输出所有文件和文件夹
for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    lowName.append(file)
    with open(filepath, 'r') as f:  # 读取文件
        data = f.readlines()
        f.close()  # 关
        # 将文件转换成字符串
        text = ""
        for line in data:
            line = line.replace("/", " or ")
            line = line.replace("(", "")
            line = line.replace(")", "")
            for word in line.split(" "):
                if word not in stopwords:
                    text += word + " "

    key = file.split('_')[0]
    key = key[1: len(key)]

    if answer.__contains__(key):
        answer.get(key).append(file)
    else:
        tmp = []
        tmp.append(file)
        answer[key] = tmp
    low.append(text)


# Tokenize input texts

high_inputs = tokenizer(high, padding=True, truncation=True, return_tensors="pt")
low_inputs = tokenizer(low, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    high_embeddings = model(**high_inputs, output_hidden_states=True, return_dict=True).pooler_output
    low_embeddings = model(**low_inputs, output_hidden_states=True, return_dict=True).pooler_output

k1set = [0.05]
k2set = [0.03]
for k1 in k1set:
    for k2 in k2set:
        print("k1: " + str(k1) + "," + "k2: " + str(k2))
        all = 0
        idf_map = {}
        dict_idf = {}
        result = {0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0, 1: 0, "pre1": 0, "pre2": 0, "pre5": 0,
                  "recall1": 0, "recall2": 0, "recall5": 0}
        result2 = {"pre1": 0, "pre2": 0, "pre5": 0,
                  "recall1": 0, "recall2": 0, "recall5": 0}
        map = 0
        query = 0
        ap = 0
        aps = []
        for i in range(0, len(high)):
            highTmp = highName[i].split('.')[0]
            highTmp = highTmp[1: len(highTmp)]

            answerSet = []
            for ans in answer.get(highTmp):
                answerSet.append(ans)

            dict = {}
            correct = 0
            label_text = {}
            for j in range(0, len(low)):
                label_text[lowName[j]] = j
                cosine_sim_0_1 = 1 - cosine(high_embeddings[i], low_embeddings[j])
                dict[lowName[j]] = cosine_sim_0_1
            list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

            curCorrect = 0
            dict = {}
            allPrecision = 0
            for j in range(0, len(list)):
                level = j
                k = j
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
                if 1 == level + 1:
                    dict["pre1"] = curCorrect / (level + 1)
                    dict["recall1"] = curCorrect / len(answerSet)
                if 2 == level + 1:
                    dict["pre2"] = curCorrect / (level + 1)
                    dict["recall2"] = curCorrect / len(answerSet)
                if 5 == level + 1:
                    dict["pre5"] = curCorrect / (level + 1)
                    dict["recall5"] = curCorrect / len(answerSet)
            result2["pre1"] = (result2.get("pre1") + dict.get("pre1", 0))
            result2["pre2"] = (result2.get("pre2") + dict.get("pre2", 0))
            result2["pre5"] = (result2.get("pre5") + dict.get("pre5", 0))
            result2["recall1"] = (result2.get("recall1") + dict.get("recall1", 0))
            result2["recall2"] = (result2.get("recall2") + dict.get("recall2", 0))
            result2["recall5"] = (result2.get("recall5") + dict.get("recall5", 0))

        result2["f1_1"] = (2 * result2["pre1"] * result2["recall1"]) / (result2["pre1"] + result2["recall1"])
        result2["f1_2"] = (2 * result2["pre2"] * result2["recall2"]) / (result2["pre2"] + result2["recall2"])
        result2["f1_5"] = (2 * result2["pre5"] * result2["recall5"]) / (result2["pre5"] + result2["recall5"])
        for key, value in result2.items():
            print(f"{key}: " + "%.2f" % (value / 17))
