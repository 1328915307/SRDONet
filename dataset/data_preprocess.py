import re
import python_speech_features
import numpy as np
import pickle
import os

import torch
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd

sample_interval= 0.01
window_len = 0.025
n_mfcc = 12
sample_delay =14
sample_len = 628
step = sample_len
Total_file = []

def extract_id_from_filename(filename):
    # 定义匹配规则的正则表达式
    pattern = re.compile(r'^read.*?_(\d+).*?_(\d+)_.*\.wav$')
    # 使用正则表达式匹配文件名
    match = pattern.match(filename)

    # 如果匹配成功，提取ID并返回；否则返回None
    if match:
        return match.group(2)
    else:
        return None

def get_ids_from_files(directory):
    # 获取目录下所有文件
    files = os.listdir(directory)

    # 仅选择以 "read" 开头的文件
    read_files = [filename for filename in files if filename.startswith('read')]

    # 用于存储提取的ID
    ids = []

    # 遍历匹配文件名并提取ID
    for filename in read_files:
        if len(filename.split('_')) == 8:
            extracted_id = extract_id_from_filename(filename)
            if extracted_id:
                ids.append(int(extracted_id))

    return set(ids)

def dataprogress(path, excelfile):
    # 读取Excel文件
    ExcelData = pd.read_excel(excelfile, sheet_name='抑郁0123')
    # 取出"健康组"和"抑郁组"两列的数据
    data = ExcelData['cust_id1'].tolist()
    Group = ExcelData['Group'].tolist()

    age = ExcelData['年龄'].tolist()
    gender = ExcelData['性别'].tolist()

    dirList = os.listdir(path)



    #抑郁depressed 焦虑anxious 压力pressure 失眠insomnia
    # 将数据和标签放入一个列表
    file_list = list(zip(data, Group))
    new_sr = 16000
    sum = 0
    for idx, item in enumerate(file_list):
        file_name = item[0]
        save_path = '/mnt/disk/caoyuhao/data/data_preprosess/depressed_audio_0123_628_39/' + str(file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(save_path)
        label = item[1]
        for audiofile in dirList:
            if len(audiofile.split('_')) == 8:
                if extract_id_from_filename(audiofile) == str(file_name):
                    # 读取音频文件
                    waveform, sr = torchaudio.load(os.path.join(path, audiofile))
                    # 初始化重采样器
                    resampler = Resample(sr, new_sr)
                    # 重采样
                    resampled_audio = resampler(waveform)
                    #边界填充
                    resampled_audio = np.insert(resampled_audio, 0, np.zeros(1920))
                    resampled_audio = np.append(resampled_audio, np.zeros(1920))
                    #mfcc特征提取
                    mfcc = python_speech_features.mfcc(resampled_audio, new_sr, winstep=sample_interval, numcep=40)
                    time_len = mfcc.shape[0]
                    length = 0
                    #mfcc切割并保存
                    for input_idx in range(int((time_len - sample_len) / step) + 1):
                        input_feat = mfcc[step * input_idx:step * input_idx + sample_len, :]
                        with open(os.path.join(save_path, str(length) + '.pkl'), 'wb') as f:
                            pickle.dump((input_feat, label), f)
                        length += 1
                    sum += 1
                    print('id为' + str(file_name) + '的人的音频已经处理完毕！')
    print('总人数为'+str(sum))

def dataprogress_justcut(path, excelfile):
    # 读取Excel文件
    ExcelData = pd.read_excel(excelfile, sheet_name='正常疾病0123')
    # 取出"健康组"和"抑郁组"两列的数据
    data = ExcelData['cust_id1'].tolist()
    Group = ExcelData['Group'].tolist()

    age = ExcelData['年龄'].tolist()
    gender = ExcelData['性别'].tolist()

    dirList = os.listdir(path)

    # 将数据和标签放入一个列表
    file_list = list(zip(data, Group))
    new_sr = 16000
    samples = int(6.28*new_sr)
    for idx, item in enumerate(file_list):
        file_name = item[0]
        save_path = '/mnt/disk/caoyuhao/data/data_preprosess/all_audio_0123_628_justcut/' + str(file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(save_path)
        label = item[1]
        for audiofile in dirList:
            if len(audiofile.split('_')) == 8:
                if extract_id_from_filename(audiofile) == str(file_name):
                    # 读取音频文件
                    waveform, sr = torchaudio.load(os.path.join(path, audiofile))
                    # 初始化重采样器
                    resampler = Resample(sr, new_sr)
                    # 重采样
                    waveform = resampler(waveform)

                    # 计算可以裁剪出的子音频数量
                    total_samples = waveform.shape[1]
                    num_sub_audios = total_samples // samples

                    # 裁剪音频并保存到列表中
                    sub_audios = []
                    for i in range(int(num_sub_audios)):
                        start = i * samples
                        end = start + samples
                        sub_audio = waveform[:, start:end]
                        sub_audios.append(sub_audio)

                    # 保存子音频到文件中
                    for i, sub_audio in enumerate(sub_audios):
                        output_path = os.path.join(save_path, f"audio_{i}_{label}.wav")
                        torchaudio.save(output_path, sub_audio, new_sr)
                        if(sub_audio.shape[1] != samples):
                            print("Shape:", sub_audio.shape[1])
                            print(output_path)

                    print('id为' + str(file_name) + '的人的音频已经处理完毕！')



def dataprogress_all(path, excelfile):
    # 读取Excel文件
    ExcelData = pd.read_excel(excelfile, sheet_name='03')
    # 取出"健康组"和"抑郁组"两列的数据
    data = ExcelData['cust_id'].tolist()
    Group = ExcelData['Group'].tolist()

    dirList = os.listdir(path)
    Total_Data, Total_label = [], []
    #抑郁depressed 焦虑anxious 压力pressure 失眠insomnia
    # 将数据和标签放入一个列表
    file_list = list(zip(data, Group))
    new_sr = 16000
    sum = 0
    for idx, item in enumerate(file_list):
        file_name = item[0]
        print(file_name)
        label = item[1]
        for audiofile in dirList:
            if len(audiofile.split('_')) == 8:
                if extract_id_from_filename(audiofile) == str(file_name) and audiofile.split('_')[3] not in Total_file:
                    # 读取音频文件
                    waveform, sr = torchaudio.load(os.path.join(path, audiofile))
                    # 初始化重采样器
                    resampler = Resample(sr, new_sr)
                    # 重采样
                    resampled_audio = resampler(waveform)
                    # 归一化音频波形到 [-1, 1] 范围
                    normalized_audio = resampled_audio / torch.max(torch.abs(resampled_audio))
                    Total_file.append(str(file_name))
                    Total_Data.append(normalized_audio)
                    if label == 0:  # group = 0 健康人
                        Total_label.append(0)
                    else:  # group = 1 病人
                        Total_label.append(1)
                    print('病人id为'+str(file_name)+'处理成功')
                    sum += 1
    print('一共处理成功' + str(sum) +'个人')
    return Total_Data, Total_label


if __name__ == '__main__':
    dirs = ['1Allsites_reading', '30<length<40/1']
    # exceltable = "/home/idal-01/data/Adolescent_voice_all/label/30-60/label/重合/两次共有人群中评分不变-classification.xlsx"
    exceltable = "/home/idal-01/caoyuhao/data/两次低分压力匹配总表.xlsx"
    # audiowavfile = "/mnt/disk/caoyuhao/data/Adolescent_voice_all"
    # rootpath = os.path.join(audiowavfile, path)
    # 切割并提取mfcc特征保存
    # dataprogress(rootpath, exceltable)
    # # 仅仅切割音频
    # dataprogress_justcut(rootpath, exceltable)
    # 不处理直接全部打包成一个pkl文件
    DATA, LABEL = [], []
    for path in dirs:
        audiowavfile = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV"
        rootpath = os.path.join(audiowavfile, path)
        Total_Data, Total_label= dataprogress_all(rootpath, exceltable)
        DATA.extend(Total_Data)
        LABEL.extend(Total_label)

    # 写入pkl文件中
    print("开始写入文件中")
    f = open('/home/idal-01/caoyuhao/data/data_preprosess_new/两次低分压力重症匹配.pkl', 'wb')
    # 将数据 放入pickle中保存
    pickle.dump((DATA, LABEL), f)
    f.close()
    print(len(Total_file))

    print("写入完毕")