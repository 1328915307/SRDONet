import re
import wave
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler


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

def dataprogress_EATD(prefix):
    all_items = os.listdir(prefix)
    Scaler = StandardScaler()
    Total_Data, Total_label = [], []
    for file_path in all_items:
        DepressionAudio = os.path.join(prefix, file_path)
        waveList = ['positive_out.wav', 'neutral_out.wav', 'negative_out.wav']
        data = np.array([], dtype=np.int16)
        # 循环读取 WAV 文件，并将数据拼接到数组中
        for wav_file in waveList:
            with wave.open(os.path.join(DepressionAudio, wav_file), 'rb') as wf:
                # 读取音频数据并拼接到数组中
                frames = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
                arr = np.frombuffer(frames, dtype=np.int16)
                data = np.concatenate((data, arr))
        length = len(data)
        data = Scaler.fit_transform(data.reshape(length, 1))
        Total_Data.append(data.squeeze())

        with open(os.path.join(DepressionAudio, 'new_label.txt'), 'r') as file:
            content = float(file.read())
        if content < 53:
            Total_label.append(0)
        else:
            Total_label.append(1)

    return Total_Data, Total_label

if __name__ == '__main__':
    # 不处理直接全部打包成一个pkl文件
    rootpath = '/home/idal-01/data/EATD-Corpus/EATD-Corpus/EATD-Corpus/'
    Total_Data, Total_label = dataprogress_EATD(rootpath)
    DATA, LABEL = [], []
    DATA.extend(Total_Data)
    LABEL.extend(Total_label)

    # 写入pkl文件中
    print("开始写入文件中")
    f = open('/home/idal-01/caoyuhao/data/data_preprosess_new/EATD.pkl', 'wb')
    # 将数据 放入pickle中保存
    pickle.dump((DATA, LABEL), f)
    f.close()
    print(len(Total_file))

    print("写入完毕")