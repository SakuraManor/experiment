import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import re
import os, sys
import operator
import math

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large", local_files_only=True)
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large", local_files_only=True)
# Tokenize input texts



# 打开文件
COOKED_FOLDER = './target/'  # 文件夹的地址
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


inputs = tokenizer(text1s, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

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

curCorrect = 0
corrects = ["66.txt", "67.txt", "68.txt", "69.txt", "70.txt", "71.txt", "72.txt",
            "73.txt", "74.txt", "75.txt", "76.txt", "77.txt", "78.txt", "79.txt"]
dict_idf = {}
for j in range(0, len(idf)):
    print(idf[j][0])
for j in range(0, len(idf)):
    len1 = len(text1s)
    tmp = idf[j][1] * math.log((len1 + 1)/(j+1))
    dict_idf[idf[j][0]] = tmp

print(dict_idf)
# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar