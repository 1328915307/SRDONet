import os
import pickle
import re

import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
import pandas as pd

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

if __name__ == '__main__':
    audiowavfile1 = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/2Allsites_reading"
    audiowavfile2 = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/30<length<40/2"
    audiowavfile3 = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/1Allsites_reading"
    audiowavfile4 = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/30<length<40/1"
    # exceltable = "/home/idal-01/caoyuhao/data/平衡后数据总表.xlsx"
    exceltable = "/home/idal-01/caoyuhao/data/平衡后数据总表.xlsx"
    result_ids1 = get_ids_from_files(audiowavfile1)
    result_ids2 = get_ids_from_files(audiowavfile2)
    result_ids3 = get_ids_from_files(audiowavfile3)
    result_ids4 = get_ids_from_files(audiowavfile4)
    # 读取Excel文件
    ExcelData = pd.read_excel(exceltable, sheet_name='失眠0123')
    # 取出"健康组"和"抑郁组"两列的数据
    data = ExcelData['cust_id1'].tolist()
    Group = ExcelData['Group'].tolist()
    file_list = list(zip(data, Group))
    result = []
    for idx, item in enumerate(file_list):
        file_name = item[0]
        label = item[1]
        num = int(file_name)
        if int(label) == 1:
            # if (num in result_ids1 or num in result_ids2) and (num not in result_ids3 and num not in result_ids4):
            #     result.append(num)
            # elif (num in result_ids3 or num in result_ids4) and (num not in result_ids1 and num not in result_ids2):
            #     result.append(num)
            if (num in result_ids1 or num in result_ids2) and (num in result_ids3 or num in result_ids4):
                result.append(num)

    print(len(result))
    # 保存结果到文件
    with open("/home/idal-01/caoyuhao/data/两次都有30语音的失眠人id.txt", "w") as file:
        for num in result:
            file.write(str(num) + "\n")

    # i = int(44213)
    # if i in result_ids1 or i in result_ids2:
    #     print("能在第二批合格数据中找到")
    # else:
    #     print("不能在第二批合格数据中找到")
    # if i in result_ids3 or i in result_ids4:
    #     print("能在第一批合格数据中找到")
    # else:
    #     print("不能在第一批合格数据中找到")
    # if i in result_ids1:
    #     print("能在第二批40-60合格数据中找到")
    # elif i in result_ids2:
    #     print("能在第二批30-40合格数据中找到")
    # if i in result_ids3:
    #     print("能在第一批40-60合格数据中找到")
    # elif i in result_ids4:
    #     print("能在第一批30-40合格数据中找到")
    # {'24947', '45499', '44299', '7690', '45961', '54922', '53048', '9718', '44507', '25589', '45957', '45591', '33018',
    #  '25245', '44953', '25993', '45779', '25325', '46974', '33162',
    # def save_to_excel(data, excel_file):
    #     df = pd.DataFrame(data)
    #     df.to_excel(excel_file, index=False)
    #
    #
    # # 指定要保存的Excel文件路径
    # excel_file = "result_ids.xlsx"
    #
    # # 将result_ids保存到Excel文件中
    # save_to_excel(result_ids, excel_file)

