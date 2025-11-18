import glob
import os
import pickle
import random
import wave
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torchaudio
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from torch import nn
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import csv
import librosa.feature
import librosa.display
import h5py


def find_duplicates(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    duplicates = unique_elements[counts > 1]
    return list(duplicates)


def dataprogress(path, excelfile):
    Total_Data, Total_label, Total_age, Total_sex = [], [],[],[]
    Total_file, Audio_files = [], []

    # 读取Excel文件
    ExcelData = pd.read_excel(excelfile, sheet_name='123')
    # 取出"健康组"和"抑郁组"两列的数据
    data = ExcelData['cust_id'].tolist()
    Group = ExcelData['Group'].tolist()
    Sex = ExcelData['性别'].tolist()
    Age = ExcelData['年龄'].tolist()
    #
    # health_data = health_data
    # depression_data = ExcelData['抑郁组'].tolist()
    #
    # # 取出"label1"和"label2"列的数据作为标签
    # health_label = ExcelData['健康组抑郁等级（0=无）'].tolist()
    # depression_label = ExcelData['抑郁等级（1=轻度；2=中度；3=重度；4=非常重度）'].tolist()
    #
    # health_score = ExcelData['健康组抑郁分数'].tolist()[0:200]
    # depression_score = ExcelData['抑郁组分数'].tolist()

    # 使用NumPy的where函数获取元素为零的下标
    # zero_indices = np.where(np.array(health_score) == 0)[0]
    # health_data = np.array(health_data)[zero_indices][0:200]
    # health_label = np.array(health_label)[zero_indices][0:200]

    dirList = os.listdir(path)

    # pathList = []cd co
    #
    # for age in ageList:
    #     path_per = os.path.join(path, age)
    #     dirList = os.listdir(path_per)
    #     for file in dirList:
    #         pathList.append(os.path.join(path_per, file))

    # 将数据和标签放入一个列表
    file_list = list(zip(data, Group,Sex, Age))
    new_sr = 16000
    Scaler = StandardScaler()
    for idx, item in enumerate(file_list):
        file_name = item[0]
        print(file_name)
        label = item[1]
        sex = item[2]
        age = item[3]
        for audiofile in dirList:
            if audiofile.split('_')[3] == str(file_name) and audiofile.split('_')[3] not in Total_file:
                wavefile = wave.open(os.path.join(path, audiofile))
                nframes = wavefile.getnframes()
                sr = wavefile.getframerate()
                each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
                resampled_audio_data = signal.resample(each_data, int(len(each_data)*new_sr/sr))
                length = len(resampled_audio_data)
                resampled_audio_data = Scaler.fit_transform(resampled_audio_data.reshape(length, 1))
                Total_file.append(str(file_name))
                Total_Data.append(resampled_audio_data.squeeze())
                if label == 0:
                    Total_label.append(0)
                else:
                    Total_label.append(1)
                Total_sex.append(sex)
                Total_age.append(age)
    find_duplicates(Total_file)
    # 写入pkl文件中
    print("开始写入文件中")
    f = open('/home/idal-01/haoyong/adolescent/1adolescent_AllInfo.pkl', 'wb')
    # 将数据 放入pickle中保存
    pickle.dump((Total_Data, Total_label,Total_age, Total_sex), f)
    f.close()
    print("写入完毕")

if __name__ == '__main__':
    # audiowavfile = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/第二次筛查/射阳"
    audiowavfile = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/第一次筛查/16KHz"
    # exceltable = "/home/idal-01/data/Adolescent_voice_all/label/第一次/first_total.xlsx"
    exceltable = "/home/idal-01/data/Adolescent_voice_all/label/30-60/label/重合/两次共有人群中评分不变-classification.xlsx"
    # Audio_files = []
    # dirList = os.listdir(audiowavfile)
    # for audiofile in dirList:
    #     if audiofile.split('_')[3]:
    #         Audio_files.append(int(audiofile.split('_')[3]))
    # Audio_files = np.unique(np.array(Audio_files))
    # df = pd.DataFrame(Audio_files)
    # df.to_excel('adolescent_audio_2.xlsx', index=False)

    dataprogress(audiowavfile, exceltable)