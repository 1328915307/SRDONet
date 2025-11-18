import re
import wave
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

sample_interval = 0.01
window_len = 0.025
n_mfcc = 12
sample_delay = 14
sample_len = 628
step = sample_len
Total_file = []

def extract_id_from_filename(filename):
    """
    从 NRAC 文件名中提取 ID
    例如：NN_00000848_S001-P.wav → NN_00000848
    （去掉 _S001，只保留人的编号）
    """
    pattern = re.compile(r'^(NN_\d+)_S\d+-P\.wav$')
    match = pattern.match(filename)
    if match:
        return match.group(1)
    else:
        return None


def dataprogress_NRAC(data_dir, excel_path, unmatched_save_path, matched_save_path):
    """
    处理 NRAC 数据集：
    1. 读取 wav 文件并匹配 Excel 中的标签
    2. 同一个人（NN_xxxxxxxx）多个语音只算一个标签
    3. diagnosis: 1 -> 健康(0)，2/4 -> 患者(1)
    4. 保存未匹配ID和匹配成功ID（加 _S001）
    """
    # === 读取标签表 ===
    df = pd.read_excel(excel_path, sheet_name='Sheet1')

    # 创建 {人ID: 标签} 映射（同一个人只保留一个标签）
    id_label_map = {}

    for _, row in df.iterrows():
        std_id = str(row['standard_id']).strip()  # 例如 NN_00000849_S001
        diag = int(row['diagnosis'])
        person_id = std_id.split('_S')[0]  # 提取人编号（NN_00000849）

        if diag == 1:
            label = 0
        elif diag in [2, 4]:
            label = 1
        else:
            continue  # 忽略未定义标签

        # 若同一人多条记录，只保留第一次出现的标签
        if person_id not in id_label_map:
            id_label_map[person_id] = label

    print(f"共读取到 {len(id_label_map)} 个有效被试标签。")

    # === 处理音频数据 ===
    Scaler = StandardScaler()
    Total_Data, Total_label = [], []

    all_files = os.listdir(data_dir)
    wav_files = [f for f in all_files if f.endswith('.wav')]

    # 每人可能有多个音频段，需要合并
    person_audio_map = defaultdict(list)

    for wav_file in wav_files:
        wav_path = os.path.join(data_dir, wav_file)
        person_id = extract_id_from_filename(wav_file)
        if not person_id:
            continue
        if person_id not in id_label_map:
            continue  # 跳过无标签的人

        person_audio_map[person_id].append(wav_path)

    print(f"匹配到 {len(person_audio_map)} 位有语音数据的被试。")

    # === 计算未匹配与已匹配ID ===
    unmatched_ids = set(id_label_map.keys()) - set(person_audio_map.keys())
    matched_ids = set(person_audio_map.keys())

    print(f"有 {len(unmatched_ids)} 个有标签但无语音数据的被试。")
    print(f"有 {len(matched_ids)} 个成功匹配的被试。")

    # 保存未匹配ID
    if len(unmatched_ids) > 0:
        with open(unmatched_save_path, 'w') as f:
            for uid in sorted(unmatched_ids):
                f.write(uid + '\n')
        print(f"未匹配的ID已保存到：{unmatched_save_path}")

    # 保存匹配成功ID（加 _S001）
    if len(matched_ids) > 0:
        with open(matched_save_path, 'w') as f:
            for uid in sorted(matched_ids):
                f.write(uid + '_S001\n')
        print(f"匹配成功的ID已保存到：{matched_save_path}")

    # === 拼接每个被试的所有语音段并标准化 ===
    for pid, wav_list in person_audio_map.items():
        data = np.array([], dtype=np.int16)
        for wav_path in wav_list:
            with wave.open(wav_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                arr = np.frombuffer(frames, dtype=np.int16)
                data = np.concatenate((data, arr))

        # 标准化
        length = len(data)
        data = Scaler.fit_transform(data.reshape(length, 1))
        Total_Data.append(data.squeeze())
        Total_label.append(id_label_map[pid])

    print(f"成功处理 {len(Total_Data)} 条被试语音样本。")
    return Total_Data, Total_label


if __name__ == '__main__':
    rootpath = '/home/idal-01/data/ward_njnk/P-6_resample'
    excel_path = '/home/idal-01/data/ward_njnk/P-6_resample抑郁总表.xlsx'
    unmatched_save_path = '/home/idal-01/caoyuhao/data/data_preprosess_new/NRAC_unmatched_ids.txt'
    matched_save_path = '/home/idal-01/caoyuhao/data/data_preprosess_new/NRAC_matched_ids.txt'

    Total_Data, Total_label = dataprogress_NRAC(rootpath, excel_path, unmatched_save_path, matched_save_path)

    DATA, LABEL = [], []
    DATA.extend(Total_Data)
    LABEL.extend(Total_label)

    # 统计健康人和病人数量
    healthy_count = sum(1 for l in LABEL if l == 0)
    patient_count = sum(1 for l in LABEL if l == 1)

    print("\n=== 数据统计 ===")
    print(f"健康人数量: {healthy_count}")
    print(f"病人数量: {patient_count}")
    print(f"总样本数: {len(LABEL)}")

    # 写入文件
    print("\n开始写入文件中...")
    save_path = '/home/idal-01/caoyuhao/data/data_preprosess_new/NRAC_P6.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump((DATA, LABEL), f)

    print(f"写入完毕，共保存 {len(DATA)} 条样本。")
