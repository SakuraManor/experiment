import re
import operator
import os, sys
from lsi_model import *


stopwords = []
# 打开文件
COOKED_FOLDER = './GanntDataset/high/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

high_content = []
high_name = []

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


dictionary, model, index = create_lsi_model(low_contents, 64)

best_value = 0
best_map = 0
# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
# result = {0.1: 0, 0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0, 1: 0, "pre1": 0, "pre2": 0, "pre5": 0,
#           "recall1": 0, "recall2": 0, "recall5": 0}
result = {"pre1": 0, "pre2": 0, "pre5": 0, "recall1": 0, "recall2": 0, "recall5": 0}
ap = 0
for i in range(0, len(high_content)):
    highTmp = high_name[i].split('.')[0]
    highTmp = highTmp[1: len(highTmp)]

    # 收集answerSet的集合
    answerSet = []
    for ans in answer.get(highTmp):
        answerSet.append(ans)


    # id = 69 + i
    correct = 0
    sims = calculate_sim_by_lsi(dictionary, model, index, high_content[i])
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

    # result[0.1] = (result.get(0.1) + cur_artifact_pr.get(0.1))
    # result[0.2] = (result.get(0.2) + cur_artifact_pr.get(0.2))
    # result[0.4] = (result.get(0.4) + cur_artifact_pr.get(0.4))
    # result[0.6] = (result.get(0.6) + cur_artifact_pr.get(0.6))
    # result[0.8] = (result.get(0.8) + cur_artifact_pr.get(0.8))
    # result[1] = (result.get(1) + cur_artifact_pr.get(1))
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
