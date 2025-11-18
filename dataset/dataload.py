# -*- coding: utf-8 -*-


import os
import random
import pickle
import numpy as np
import cv2
import python_speech_features
import torch
import torch.utils.data as data
import torchaudio
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader, Dataset
import librosa
import time
import copy

from adolescent_test.untils import data_segmentwd, datafft


#加载的两个语音mfcc特征用于借口任务
class MyDataset(data.Dataset):
    def __init__(self, dataset_dir, mode, train_ratio, seed=1):
        self.data_path = dataset_dir
        self.mode = mode
        self.train_ratio = train_ratio
        self.seed = seed
        # 获取所有子文件夹名
        all_persons = [f.name for f in os.scandir(self.data_path) if f.is_dir()]
        # (按人划分训练测试集)
        # 随机打乱子文件夹顺序
        random.seed(seed)
        random.shuffle(all_persons)

        # 计算划分点
        split_point = int(train_ratio * len(all_persons))

        # 根据 mode 选择是训练集还是测试集
        if mode == 'train':
            self.persons = all_persons[:split_point]
        elif mode == 'test':
            self.persons = all_persons[split_point:]

        self.samples = []
        for person in self.persons:
            fragments = [f for f in os.listdir(os.path.join(self.data_path, str(person))) if f.endswith('.pkl')]
            random.seed(seed)
            random.shuffle(fragments)
            self.samples.extend([(person, fragment) for fragment in fragments])


        # # (按片段划分训练测试)遍历每个子文件夹，随机划分训练集和测试集
        # self.samples = []
        # for person in self.persons:
        #     fragments = [f for f in os.listdir(person) if f.endswith('.pkl')]
        #     random.seed(seed)
        #     random.shuffle(fragments)
        #
        #     if mode == 'train':
        #         selected_fragments = fragments[:int(len(fragments) * train_ratio)]
        #     elif mode == 'test':
        #         selected_fragments = fragments[int(len(fragments) * train_ratio):]
        #
        #     self.samples.extend([(person, fragment) for fragment in selected_fragments])



    def __getitem__(self, index):
        # 按片段划分
        # # 随机选择一个 person
        # person, fragment1 = self.samples[index]
        # # 随机选择该 person 的另一个语音片段
        # fragment2 = random.choice([f for p, f in self.samples if p == person and f != fragment1])
        # # 完整路径
        # audio_path1 = os.path.join(self.data_path, str(person), fragment1)
        # audio_path2 = os.path.join(self.data_path, str(person), fragment2)
        #按人划分
        person, fragment1 = self.samples[index]
        # 随机选择该 person 的另一个语音片段
        all_fragment = [file for file in os.listdir(os.path.join(self.data_path, str(person))) if file.endswith('.pkl') and file != fragment1]
        fragment2 = random.choice(all_fragment)
        audio_path1 = os.path.join(self.data_path, str(person), fragment1)
        audio_path2 = os.path.join(self.data_path, str(person), fragment2)


        f=open(audio_path1, 'rb')
        mfcc1, label1 = pickle.load(f)
        mfcc1 = torch.FloatTensor(mfcc1[:, 1:])
        f.close()

        f=open(audio_path2, 'rb')
        mfcc2, label2 = pickle.load(f)
        mfcc2 = torch.FloatTensor(mfcc2[:, 1:])
        f.close()

        mfcc1 = torch.unsqueeze(mfcc1, 0).cuda()
        mfcc2 = torch.unsqueeze(mfcc2, 0).cuda()


        target1 = mfcc1.detach().clone()

        target2 = mfcc2.detach().clone()

        label1 = torch.tensor(label1).long().cuda()
        label2 = torch.tensor(label2).long().cuda()

        return {"input1": mfcc1, "target1": target1,
                "input2": mfcc2, "target2": target2,
                "label1": label1,  "label2": label2,
              }


    def __len__(self):
        # 按片段划分
        return  len(self.samples)



#加载的两个语音片段wav特征用于借口任务
class MyDataset_two_audio(data.Dataset):
    def __init__(self, dataset_dir, mode, train_ratio, seed=1):
        self.data_path = dataset_dir
        self.mode = mode
        self.train_ratio = train_ratio
        self.seed = seed
        # 获取所有子文件夹名
        all_persons = [f.name for f in os.scandir(self.data_path) if f.is_dir()]
        # (按人划分训练测试集)
        # 随机打乱子文件夹顺序
        random.seed(seed)
        random.shuffle(all_persons)

        # 计算划分点
        split_point = int(train_ratio * len(all_persons))

        # 根据 mode 选择是训练集还是测试集
        if mode == 'train':
            self.persons = all_persons[:split_point]
        elif mode == 'test':
            self.persons = all_persons[split_point:]

        self.samples = []
        for person in self.persons:
            fragments = [f for f in os.listdir(os.path.join(self.data_path, str(person))) if f.endswith('.wav')]
            file_name = os.path.basename(fragments[0])
            label = file_name[-5]
            random.seed(seed)
            random.shuffle(fragments)
            self.samples.extend([(person, fragment, label) for fragment in fragments])



    def __getitem__(self, index):
        #按人划分
        person, fragment1, label = self.samples[index]
        # 随机选择该 person 的另一个语音片段
        all_fragment = [file for file in os.listdir(os.path.join(self.data_path, str(person))) if file.endswith('.wav') and file != fragment1]
        fragment2 = random.choice(all_fragment)
        audio_path1 = os.path.join(self.data_path, str(person), fragment1)
        audio_path2 = os.path.join(self.data_path, str(person), fragment2)

        waveform1, sr1 = torchaudio.load(audio_path1)
        waveform1 = np.insert(waveform1, 0, np.zeros(80))
        waveform1 = np.append(waveform1, np.zeros(80))
        mfcc1 = python_speech_features.mfcc(waveform1, samplerate=16000, winstep=0.01)
        mfcc1 = torch.FloatTensor(mfcc1[:, 1:])

        waveform2, sr2 = torchaudio.load(audio_path2)
        waveform2 = np.insert(waveform2, 0, np.zeros(80))
        waveform2 = np.append(waveform2, np.zeros(80))
        mfcc2 = python_speech_features.mfcc(waveform2, samplerate=16000, winstep=0.01)
        mfcc2 = torch.FloatTensor(mfcc2[:, 1:])

        mfcc1 = torch.unsqueeze(mfcc1, 0).cuda()
        mfcc2 = torch.unsqueeze(mfcc2, 0).cuda()


        target1 = mfcc1.detach().clone()

        target2 = mfcc2.detach().clone()

        label = int(label)
        label1 = torch.tensor(label).long().cuda()
        label2 = torch.tensor(label).long().cuda()

        return {"input1": waveform1, "target1": target1,
                "input2": waveform2, "target2": target2,
                "label1": label1,  "label2": label2,
              }


    def __len__(self):
        # 按片段划分
        return  len(self.samples)





#加载的mfcc特征
class MyDatasetone(data.Dataset):
    def __init__(self, dataset_dir, mode, train_ratio, seed=1):
        self.data_path = dataset_dir
        self.mode = mode
        self.train_ratio = train_ratio
        self.seed = seed
        # 获取所有子文件夹名
        all_persons = [f.name for f in os.scandir(self.data_path) if f.is_dir()]
        # (按人划分训练测试集)
        # 随机打乱子文件夹顺序
        random.seed(seed)
        random.shuffle(all_persons)

        # 计算划分点
        split_point = int(train_ratio * len(all_persons))

        # 根据 mode 选择是训练集还是测试集
        if mode == 'train':
            self.persons = all_persons[:split_point]
        elif mode == 'test':
            self.persons = all_persons[split_point:]

        self.samples = []
        for person in self.persons:
            fragments = [f for f in os.listdir(os.path.join(self.data_path, str(person))) if f.endswith('.pkl')]
            random.seed(seed)
            random.shuffle(fragments)
            self.samples.extend([(person, fragment) for fragment in fragments])


        # # (按片段划分训练测试)遍历每个子文件夹，随机划分训练集和测试集
        # self.samples = []
        # for person in self.persons:
        #     fragments = [f for f in os.listdir(person) if f.endswith('.pkl')]
        #     random.seed(seed)
        #     random.shuffle(fragments)
        #
        #     if mode == 'train':
        #         selected_fragments = fragments[:int(len(fragments) * train_ratio)]
        #     elif mode == 'test':
        #         selected_fragments = fragments[int(len(fragments) * train_ratio):]
        #
        #     self.samples.extend([(person, fragment) for fragment in selected_fragments])



    def __getitem__(self, index):
        # 按片段划分
        # # 随机选择一个 person
        # person, fragment1 = self.samples[index]
        # # 随机选择该 person 的另一个语音片段
        # fragment2 = random.choice([f for p, f in self.samples if p == person and f != fragment1])
        # # 完整路径
        # audio_path1 = os.path.join(self.data_path, str(person), fragment1)
        # audio_path2 = os.path.join(self.data_path, str(person), fragment2)
        #按人划分
        person, fragment1 = self.samples[index]
        audio_path1 = os.path.join(self.data_path, str(person), fragment1)
        f=open(audio_path1, 'rb')
        mfcc1, label1 = pickle.load(f)
        mfcc1 = torch.FloatTensor(mfcc1[:, 1:])
        f.close()
        mfcc1 = torch.unsqueeze(mfcc1, 0).cuda()
        label1 = torch.tensor(label1).long().cuda()

        return {"input1": mfcc1,"label1": label1}


    def __len__(self):
        # 按片段划分
        return  len(self.samples)


#加载的切割后的语音wav文件
class MyDataset_audio(data.Dataset):
    def __init__(self, dataset_dir, mode, train_ratio, seed=1):
        self.data_path = dataset_dir
        self.mode = mode
        self.train_ratio = train_ratio
        self.seed = seed
        # 获取所有子文件夹名
        all_persons = [f.name for f in os.scandir(self.data_path) if f.is_dir()]
        # (按人划分训练测试集)
        # 随机打乱子文件夹顺序
        random.seed(seed)
        random.shuffle(all_persons)

        # 计算划分点
        split_point = int(train_ratio * len(all_persons))

        # 根据 mode 选择是训练集还是测试集
        if mode == 'train':
            self.persons = all_persons[:split_point]
        elif mode == 'test':
            self.persons = all_persons[split_point:]

        self.samples = []
        for person in self.persons:
            fragments = [f for f in os.listdir(os.path.join(self.data_path, str(person))) if f.endswith('.wav')]
            file_name = os.path.basename(fragments[0])
            label = file_name[-5]
            random.seed(seed)
            random.shuffle(fragments)
            self.samples.extend([(person, fragment, label) for fragment in fragments])




    def __getitem__(self, index):
        #按人划分
        person, fragment, label = self.samples[index]
        audio_path = os.path.join(self.data_path, str(person), fragment)
        waveform, sr = torchaudio.load(audio_path)

        return {"input1": waveform, "label1": label, "path": audio_path}


    def __len__(self):
        # 按片段划分
        return  len(self.samples)


def Get_allperson(data_path):
    Total_label = []
    all_persons = [f.name for f in os.scandir(data_path) if f.is_dir()]
    for person in all_persons:
        audio_path = os.path.join(data_path, str(person), '0.pkl')
        f=open(audio_path, 'rb')
        _, label = pickle.load(f)
        f.close()
        label = torch.tensor(label).long().cuda()
        Total_label.append(label)
    return all_persons, Total_label

def Get_mfccdata(person,data_path):
    Total_Data = []
    fragments = [f for f in os.listdir(os.path.join(data_path, str(person))) if f.endswith('.pkl')]
    for fragment in fragments:
        audio_path = os.path.join(data_path, str(person), fragment)
        f=open(audio_path, 'rb')
        mfcc, _ = pickle.load(f)
        mfcc = torch.FloatTensor(mfcc[:, 1:])
        f.close()
        mfcc = torch.unsqueeze(mfcc, 0)
        Total_Data.append(mfcc)
    data = np.array(Total_Data)

    return np.array(data)

def total_dataloader(config):
    Persons, Total_label = Get_allperson(config.dataset_dir)
    zip_list = list(zip(Persons, Total_label))
    random.Random(123).shuffle(zip_list)
    personList, labelList = zip(*zip_list)
    trainList_dataloader, testList_dataloader, valList_dataloader= [], [], []
    kf = StratifiedKFold(n_splits=config.KFold, shuffle=True, random_state=1)
    # kf = KFold(n_splits=config.KFold, shuffle=True, random_state=123)

    labelList = [tensor.cpu() for tensor in labelList]
    personList = np.array(personList)
    labelList = np.array(labelList)
    toPersonList = []
    for train_index, test_index in kf.split(personList, labelList):

        X_train, test_person = personList[train_index], personList[test_index]
        y_train, test_labels = labelList[train_index], labelList[test_index]
        tv_folder = StratifiedKFold(n_splits=config.KFold, random_state=1, shuffle=True).split(X_train, y_train)
        # 划分验证集
        for t_idx, v_idx in tv_folder:
            train_person, train_labels = X_train[t_idx], y_train[t_idx]
            val_person, val_labels = X_train[v_idx], y_train[v_idx]

        # 构造训练集
        train_dataset = []
        train_labels_dataset = []
        for idx, person in enumerate(train_person):
                data = Get_mfccdata(person, config.dataset_dir)
                m = data.shape[0]
                label = train_labels[idx]
                train_labels_dataset.extend(label.repeat(m))
                train_dataset.extend(data)
        # 构造验证集
        val_dataset = []
        val_labels_dataset = []
        for idx, person in enumerate(val_person):
                data = Get_mfccdata(person, config.dataset_dir)
                m = data.shape[0]
                label = val_labels[idx]
                val_labels_dataset.extend(label.repeat(m))
                val_dataset.extend(data)

        # 构造测试集
        test_dataset = []
        test_labels_dataset = []
        demo_per = []
        for idx, person in enumerate(test_person):
            data = Get_mfccdata(person, config.dataset_dir)
            m = data.shape[0]
            label = test_labels[idx]
            test_labels_dataset.extend(label.repeat(m))
            test_dataset.extend(data)
            demo_per.append(m)

        toPersonList.append(demo_per)

        train_dataset = datasets(train_dataset, train_labels_dataset)
        test_dataset = datasets(test_dataset, test_labels_dataset)
        val_dataset = datasets(val_dataset, val_labels_dataset)

        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
        test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        trainList_dataloader.append(train_dataset_loader)
        testList_dataloader.append(test_dataset_loader)
        valList_dataloader.append(val_dataset_loader)

    return trainList_dataloader, testList_dataloader, valList_dataloader, toPersonList


class datasets(Dataset):
    def __init__(self, x, label):
        self.labels = label
        self.x = x

    def __getitem__(self, idx):
        return_dic = {"input1": self.x[idx], "label1": self.labels[idx]}
        return return_dic

    def __len__(self):
        return len(self.labels)


class dataset_wav_mel(Dataset):
    def __init__(self, Total_Data, args):
        self.sample_rate = args.fs
        self.NFFT = args.NFFT
        self.nfilt = args.filter_num

        random.Random(1234).shuffle(Total_Data)
        seg_level = args.audio_length
        stride = int(seg_level * 1)

        # 构造训练集
        train_dataset = []

        for idx, data in enumerate(Total_Data):
            if len(data) / 16000 > seg_level:
                data_temp = data_segmentwd(data, int(16000 * seg_level), 16000 * stride)
                train_dataset.append(data_temp)

        self.samples = []
        for person in train_dataset:
            self.samples.extend([(person, id) for id, fragment in enumerate(person)])



    def __getitem__(self, index):
        #按人划分
        person, id = self.samples[index]
        # 创建一个列表，包含所有可能的索引
        all_indices = list(range(len(person)))
        # 从 all_indices 列表中去除给定的 id 索引
        all_indices.remove(id)
        # 从剩余的索引中随机选择一个索引
        random_id = random.choice(all_indices)
        # 获取随机选择的索引对应的同一个人的另一段语音
        waveform1 = person[random_id]
        waveform2 = person[id]
        #生成mel
        mel1 = pre_datafft(np.array(waveform1), self.sample_rate, self.NFFT, self.nfilt)
        mel2 = pre_datafft(np.array(waveform2), self.sample_rate, self.NFFT, self.nfilt)

        return {"wav1": waveform1, "wav2": waveform2,
                "mel1": mel1, "mel2": mel2,
              }
    def __len__(self):
        # 按片段划分
        return  len(self.samples)


def pre_dataloader(args, Total_Data):



    train_dataset = dataset_wav_mel(Total_Data, args)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       drop_last=True)
    return train_dataset_loader

def pre_datafft(signal, sample_rate, NFFT, nfilt):
    win_length, hop_length = int(0.025 * sample_rate), int(0.01 * sample_rate)
    # 计算短时傅里叶变换(STFT)
    window = torch.hann_window(win_length)
    signal = signal.astype(np.float32)
    stft = torch.stft(input=torch.from_numpy(signal).float(), n_fft=NFFT, window=window, hop_length=hop_length,
                      win_length=win_length, normalized=True, center=False, return_complex=True)
    # 计算功率谱
    # mag_frames = np.abs(stft)
    power_spec = torch.square(torch.abs(stft))
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)
    for i in range(1, nfilt + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
    fbank = torch.from_numpy(fbank).to(power_spec.device)

    pow_frames = power_spec.double()
    fbank = fbank.double()

    # 执行矩阵乘法，并将结果转换回float类型
    filter_banks = torch.matmul(pow_frames.permute(1, 0), fbank.permute(1, 0)).permute(1, 0).float()
    filter_banks = torch.where(filter_banks == 0, torch.finfo(torch.float).eps, filter_banks)

    filter_banks = 20 * torch.log10(filter_banks)

    return filter_banks