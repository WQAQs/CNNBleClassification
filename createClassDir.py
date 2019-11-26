import os
import shutil
import globalConfig

# 每个类创建一个文件夹并将其放入文件夹中的代码
data_path = ".\\raw_data\\train"  # 原始数据存放的文件夹，通常无需改动

fileList = os.listdir(data_path)
os.chdir(data_path)
classCount = 0
for file in fileList:
    if os.path.isfile(file):
        os.mkdir(str(classCount))
        shutil.move(file, str(classCount))
        classCount += 1

