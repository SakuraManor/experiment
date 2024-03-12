import re
import os, sys

# 打开文件
COOKED_FOLDER = './source/'  # 文件夹的地址
dirs = os.listdir(COOKED_FOLDER)

code = []
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
    print(text)