import os
import shutil

import librosa
import pandas as pd
import numpy as np

# 读取 WAV 文件
def load_wav_file(file_path):
    _, sample_rate = librosa.load(file_path)
    return sample_rate

def get_file_list(directory):
    file_list = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_path[-3:] != "csv":
            file_list.append(file_name)
    return file_list

def check_duplicate_values(matrix):
    coverlist = []
    seen_values = set()
    for row in matrix:
        for value_temp in row:
            value_split = value_temp.split('_')
            if len(value_split) == 8:
                value = value_split[3]
                if value != "":
                    if value in seen_values:
                        coverlist.append(value_temp)
                    else:
                        seen_values.add(value)
    return seen_values

def check_duplicate_values2(matrix):
    # filepath = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/1Allsites_reading/"
    # matrix = get_file_list(filepath)
    coverlist = {}
    # seen_values = set()
    for row in matrix:
        value_split = row.split('_')
        if len(value_split) == 8:
            value = value_split[3]
            if value != "":
                templist = []
                templist.append(row)
                if value in coverlist:
                    coverlist[value].append(row)
                else:
                    coverlist[value] = templist

    result = {key: value for key, value in coverlist.items() if len(value) > 1}

    return result

def move_coverfile(cover_list, filepath):
    # filepath = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/1Allsites_reading/"
    for key, value in cover_list.items():
        for i in range(len(value)):
            name = os.path.join(filepath, value[i])
            target_folder = '/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/duplicate_files/1/'
            # 移动文件
            shutil.move(name, target_folder)


def delete_cover():
    # 读取三个Excel文件
    excel1 = "/home/idal-01/haoyong/adolescent/第一次_射阳_read.xlsx"
    excel2 = "/home/idal-01/haoyong/adolescent/第一次_泰州_read.xlsx"
    excel3 = "/home/idal-01/haoyong/adolescent/第一次_宜兴_read.xlsx"
    excel_all = "/home/idal-01/haoyong/adolescent/第一次_adolescent_read.xlsx"
    df1 = pd.read_excel(excel1, header=None)
    df2 = pd.read_excel(excel2, header=None)
    df3 = pd.read_excel(excel3, header=None)
    df_all = pd.read_excel(excel_all, header=None)

    print("射阳前:",len(df1))
    print("泰州前:", len(df2))
    print("宜兴前:", len(df3))
    print("总前:", len(df_all))

    # 将每个文件的数据转换为集合
    set1 = set(df1[0][1:])
    set2 = set(df2[0][1:])
    set3 = set(df3[0][1:])

    # 找到三个文件中的重复值
    duplicates = list(set1.intersection(set2)) + list(set1.intersection(set3)) + list(set2.intersection(set3))
    duplicates = list(set(duplicates))

    # 从每个文件中删除重复值
    df1 = df1[~df1[0].isin(duplicates)]
    df2 = df2[~df2[0].isin(duplicates)]
    df3 = df3[~df3[0].isin(duplicates)]
    df_all = df_all[~df_all[0].isin(duplicates)]

    print("射阳后:", len(df1))
    print("泰州后:", len(df2))
    print("宜兴后:", len(df3))
    print("总后:", len(df_all))

    # 将结果保存回原始文件
    df1.to_excel(excel1, index=False, header=False)
    df2.to_excel(excel2, index=False, header=False)
    df3.to_excel(excel3, index=False, header=False)
    df_all.to_excel(excel_all, index=False, header=False)

def select_read_video():
    # 指定目录
    directory = '/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/第二次筛查/'

    # 获取文件列表
    files = get_file_list(directory)
    result = [] #三个地方的read文件名
    for filelist in files:
        if  filelist == "泰州" or filelist == "射阳" or filelist == "宜兴":
            path = os.path.join(directory,filelist)
            wav_name_list = get_file_list(path)
            temp = [value for value in wav_name_list if "read_video" in value]
            result.append(temp)

    only_values = check_duplicate_values(result)
    print(len(only_values))
    # 将集合转换为数据框（DataFrame）
    df = pd.DataFrame(only_values, columns=["Read_ID"])

    # 将数据框写入Excel文件
    df.to_excel("第二次_adolescent_read.xlsx", index=False)
    # print()

def pass_silence():
    filepath = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/1Allsites_reading/"
    file_list = get_file_list(filepath)
    silence_segments = [] #返回有空白的语音名称
    for i in range(len(file_list)):
        filename = os.path.join(filepath,file_list[i])
        audio_data, sample_rate = librosa.load(filename)

        # 将音频数据划分为帧
        frame_length = 4096
        hop_length = 2048
        # frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)

        # 计算每个帧的能量
        frame_energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]

        # 计算数组的 L2 范数
        norm = np.linalg.norm(frame_energy)

        # 归一化数组
        frame_energy = frame_energy / norm
        # 设置能量阈值
        energy_threshold = 5e-10
        # 检测静音段
        segment_start = None

        for j, energy in enumerate(frame_energy):
            if energy < energy_threshold and segment_start is None:
                segment_start = j
            elif energy >= energy_threshold and segment_start is not None:
                segment_end = j
                segment_duration = (segment_end - segment_start) * (hop_length / sample_rate)
                if segment_duration > 1:
                    silence_segments.append(file_list[i])
                    break;
                segment_start = None

    return silence_segments

def creat_excel_from_wav():
    filepath = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/2Allsites_reading/"
    file_list = get_file_list(filepath)
    result_list = []
    for filename in file_list:
        value_split = filename.split('_')
        if len(value_split) == 8:
            value = value_split[3]
            if value != "":
                result_list.append(value)
    df = pd.DataFrame(result_list)
    df.to_excel('Allsites_reading_2.xlsx', index=False)

def creat_label_excel():
    id_path = "/home/idal-01/haoyong/adolescent/excel_all/30-60/adolescent_audio_2.xlsx"
    label_path = "/home/idal-01/haoyong/adolescent/excel_all/label1214/第二次/泰州/第二批泰州.xlsx"
    save_path = "/home/idal-01/haoyong/adolescent/excel_all/final_label_3060/第二次/泰州/第二批泰州.xlsx"
    id_list = pd.read_excel(id_path)
    label_sheet_list = pd.read_excel(label_path, sheet_name=None)
    id_list = id_list.iloc[:, 0].tolist()

    excel_writer = pd.ExcelWriter(save_path)
    for sheet in label_sheet_list:
        df = label_sheet_list[sheet]

        new_sheet = df[df.iloc[:, 0].isin(id_list)]
        new_sheet.to_excel(excel_writer, sheet_name=sheet, index=False)

    excel_writer.save()

def creat_label_excel2():
    label_path = "/home/idal-01/haoyong/adolescent/excel_all/40-60/adolescent_audio_1_4060.xlsx"
    id_path = "/home/idal-01/haoyong/adolescent/excel_all/20220901第一批所有人.xlsx"
    save_path = "/home/idal-01/haoyong/adolescent/excel_all/final_label_4060/第一次.xlsx"
    id_sheet_list = pd.read_excel(id_path, sheet_name=None)
    id_list = []
    for sheet in id_sheet_list:
        df = id_sheet_list[sheet]
        id_list.extend(df.iloc[:, 0].tolist())
    # id_list.
    label_sheet_list = pd.read_excel(label_path, sheet_name=None)

    excel_writer = pd.ExcelWriter(save_path)
    for sheet2 in label_sheet_list:
        df = label_sheet_list[sheet2]

        new_sheet = df[df.iloc[:, 0].isin(id_list)]
        new_sheet.to_excel(excel_writer, sheet_name=sheet2, index=False)

    excel_writer.save()

def FScover_num():
    num_path1 = "/home/idal-01/haoyong/adolescent/excel_all/final_label_3060/第一次.xlsx"
    num_path2 = "/home/idal-01/haoyong/adolescent/excel_all/final_label_3060/第二次.xlsx"
    id_sheet_list1 = pd.read_excel(num_path1, sheet_name=None)
    id_sheet_list2 = pd.read_excel(num_path2, sheet_name=None)
    id_list1 = []
    id_list2 = []
    for sheet1 in id_sheet_list1:
        df = id_sheet_list1[sheet1]
        id_list1.extend(df.iloc[:, 0].tolist())
    for sheet2 in id_sheet_list2:
        df2 = id_sheet_list2[sheet2]
        id_list2.extend(df2.iloc[:, 0].tolist())

    set1 = set(id_list1)
    set2 = set(id_list2)

    # 计算交集
    intersection = set1.intersection(set2)

    # 计算交集的长度
    count = len(intersection)
    print(count)


if __name__ == '__main__':
    FScover_num()



