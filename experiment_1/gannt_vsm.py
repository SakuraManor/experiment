from vsm_model import *
import re
import operator
import os, sys
import math
from gensim.models.doc2vec import Doc2Vec, TaggedDocument




# Using the pre-trained word2vec model trained using Google news corpus of 3 billion running words.
# The model can be downloaded here: https://bit.ly/w2vgdrive (~1.4GB)
# Feel free to use to your own model.




stopwords = []
# 打开文件
COOKED_FOLDER = './GanntDataset/high/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

high_content = []
high_name = []
all_content = []

# 记录所有的high
for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    high_name.append(file)
    with open(filepath, 'r') as f:  # 读取文件
        data = f.readlines()
        f.close()  # 关
        # 将文件转换成字符串
        text = ""
        for line in data:
            line = line.replace("/", " or ")
            line = line.replace("(", "")
            line = line.replace(")", "")
            text += line
        text = re.sub(r'\s+', ' ', text)
        text = re.sub('\s+', ' ', text)

    high_content.append(text)
    all_content.append(text)


# 打开文件
COOKED_FOLDER = './GanntDataset/low/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

low_contents = []
low_names = []
answer = {}
# 输出所有文件和文件夹
for file in dirs:
    filepath = COOKED_FOLDER + file  # 文件所在地址
    low_names.append(file)
    with open(filepath, 'r') as f:  # 读取文件
        data = f.readlines()
        f.close()  # 关
        # 将文件转换成字符串
        text = ""
        for line in data:
            line = line.replace("/", " or ")
            line = line.replace("(", "")
            line = line.replace(")", "")
            text += line
        text = re.sub(r'\s+', ' ', text)
        text = re.sub('\s+', ' ', text)

    key = file.split('_')[0]
    key = key[1: len(key)]

    if answer.__contains__(key):
        answer.get(key).append(file)
    else:
        tmp = []
        tmp.append(file)
        answer[key] = tmp
    low_contents.append(text)
    all_content.append(text)

dictionary, model, index = create_vsm_model(low_contents)



# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
result = {"pre1": 0, "pre2": 0, "pre5": 0,
          "recall1": 0, "recall2": 0, "recall5": 0}
ap = 0
for i in range(0, len(high_content)):
    highTmp = high_name[i].split('.')[0]
    highTmp = highTmp[1: len(highTmp)]

    # 收集answerSet的集合
    answerSet = []
    for ans in answer.get(highTmp):
        answerSet.append(ans)

    correct = 0
    sims = calculate_sim_by_vsm(dictionary, model, index, high_content[i])
    sim_dict = {}
    for j in range(0, len(sims)):
        sim_dict[low_names[j]] = sims[j]

    sim_list = sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)
    curCorrect = 0


    cur_artifact_pr = {}
    allPrecision = 0
    for j in range(0, len(sim_list)):
        level = j
        if answerSet.__contains__(sim_list[j][0]):
            curCorrect = curCorrect + 1
            curPrecision = curCorrect / (j + 1)
            allPrecision += curPrecision
        if j + 1 == 1:
            cur_artifact_pr["pre1"] = curCorrect / (j + 1)
            cur_artifact_pr["recall1"] = curCorrect / len(answerSet)
        if j + 1 == 2:
            cur_artifact_pr["pre2"] = curCorrect / (j + 1)
            cur_artifact_pr["recall2"] = curCorrect / len(answerSet)
        if j + 1 == 5:
            cur_artifact_pr["pre5"] = curCorrect / (j + 1)
            cur_artifact_pr["recall5"] = curCorrect / len(answerSet)

    result["pre1"] = (result.get("pre1") + cur_artifact_pr.get("pre1"))
    result["pre2"] = (result.get("pre2") + cur_artifact_pr.get("pre2", 0))
    result["pre5"] = (result.get("pre5") + cur_artifact_pr.get("pre5", 0))
    result["recall5"] = (result.get("recall5") + cur_artifact_pr.get("recall5", 0))
    result["recall2"] = (result.get("recall2") + cur_artifact_pr.get("recall2", 0))
    result["recall1"] = (result.get("recall1") + cur_artifact_pr.get("recall1", 0))


result["f1_1"] = (2 * result["pre1"] * result["recall1"]) / (result["pre1"] + result["recall1"])
result["f1_2"] = (2 * result["pre2"] * result["recall2"]) / (result["pre2"] + result["recall2"])
result["f1_5"] = (2 * result["pre5"] * result["recall5"]) / (result["pre5"] + result["recall5"])
for key, value in result.items():
    print(f"{key}: " + "%.2f" % (value / (len(high_content))))