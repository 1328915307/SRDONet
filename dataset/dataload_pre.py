# -*- coding: utf-8 -*-



import random
import numpy as np
import torch
import torch.utils.data as data

from torch.utils.data import DataLoader, Dataset
from adolescent_test.untils import data_segmentwd, datafft






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
        print(len(person))
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
        target1 = mel1.detach().clone()
        target2 = mel2.detach().clone()

        return {"wav1": waveform1, "wav1": waveform2,
                "mel1": mel1, "mel1": mel2,
                "traget1": target1,  "traget1": target2,
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