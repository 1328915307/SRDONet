import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from adolescent_test.model_main import E_encoder
from adolescent_test.untils import plot_mel_spectrogram
from models.Classifler.models.layers import Squeeze
from models.models_emo_con import Conv2dLSTM_Audio, orthogonal_loss
from models.ops import *
import torchvision.models as models
import functools
from models.Classifler.models.ResNet import ResNet

from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoderLayer



def gaussian_regularization_loss(output, mu, sigma):
    # 计算特征的均值和标准差
    output_mean = torch.mean(output, dim=0)
    output_std = torch.std(output, dim=0)

    # 计算 KL 散度或者均方差
    kl_loss = torch.mean(torch.pow(output_mean - mu, 2) + torch.pow(output_std - sigma, 2))

    return kl_loss

def sim_loss(p, z):  # negative cosine similarity
    return - F.cosine_similarity(p, z, dim=-1).mean()

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        # batch_size = input1.size(0)
        # input1 = input1.contiguous().view(batch_size, -1)
        # input2 = input2.contiguous().view(batch_size, -1)
        #
        # # # Zero mean
        # input1_mean = torch.mean(input1, dim=0, keepdims=True)
        # input2_mean = torch.mean(input2, dim=0, keepdims=True)
        # input1 = input1 - input1_mean
        # input2 = input2 - input2_mean
        # #
        # # # norm
        # input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        # input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        #
        # input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        # input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        #
        # diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        batch_size = input1.size(0)
        input1_flat = input1.reshape(batch_size, -1)  # 将输入扁平化，保持批次维度不变
        input2_flat = input2.reshape(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1_flat, dim=1, keepdim=True)
        input2_mean = torch.mean(input2_flat, dim=1, keepdim=True)
        input1_zero_mean = input1_flat - input1_mean
        input2_zero_mean = input2_flat - input2_mean

        # L2 normalization
        input1_l2_norm = torch.norm(input1_zero_mean, p=2, dim=1, keepdim=True)
        input1_l2 = torch.div(input1_zero_mean, input1_l2_norm + 1e-6)

        input2_l2_norm = torch.norm(input2_zero_mean, p=2, dim=1, keepdim=True)
        input2_l2 = torch.div(input2_zero_mean, input2_l2_norm + 1e-6)

        # Compute cosine similarity
        cos_sim = torch.matmul(input1_l2, input2_l2.t()).pow(2)  # 计算余弦相似度

        diff_loss = torch.mean(cos_sim)  # 计算损失

        return diff_loss


def off_diagonal(x):
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class SimLoss(nn.Module):

    def __init__(self, lambd: float = 5e-3):
        super(SimLoss, self).__init__()
        self.lambd = lambd
    def forward(self, input1, input2):
        # empirical cross-correlation matrix
        c = self.bn(input1).T @ self.bn(input2)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class AutoEncoder3(nn.Module):
    def __init__(self,config):
        super(AutoEncoder3, self).__init__()
        self.emo_encoder = Conv2dLSTM_Audio(config)
        self.con_encoder = Wav2vecNet_con(config)
        self.spe_encoder = Wav2vecNet_spe(config)
        # 冻结 con_encoder 的参数
        for param in self.con_encoder.parameters():
            param.requires_grad = False

        # 冻结 spe_encoder 的参数
        for param in self.spe_encoder.parameters():
            param.requires_grad = False

        self.classify = Classify()
        self.decoder = Decoder()
        self.dropout = nn.Dropout(p=config.dropout)

        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.diff_loss = DiffLoss()
        self.space_loss = CMD()
        self.l1loss = nn.L1Loss()
        self.labels_name = ['label1', 'label2']
        self.inputs_name = ['input1', 'input2']
        self.targets_name = ['target1', 'target2']
        self.optimizer = torch.optim.Adam(list(self.emo_encoder.parameters())
                                            +list(self.decoder.parameters())
                                            +list(self.classify.parameters()), config.lr,betas=(config.beta1, config.beta2))



    def cross(self, audio1, audio2, mfcc1, mfcc2):
        c1 = self.con_encoder(audio1)
        s1 = self.spe_encoder(audio1)
        e1 = self.emo_encoder(mfcc1)

        c2 = self.con_encoder(audio2)
        s2 = self.spe_encoder(audio2)
        e2 = self.emo_encoder(mfcc2)

        self_recon1 = self.decoder(e1, c1, s1)
        recon1 = self.decoder(e2, c1, s1)

        self_recon2 = self.decoder(e2, c2, s2)
        recon2 = self.decoder(e1, c2, s2)

        return self_recon1, recon1, self_recon2, recon2, e1, c1, s1, e2, c2, s2

    def compute_acc(self,input_label, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
        return acc

    def process(self, data):

        labels = [data[name] for name in self.labels_name]
        inputs = [data[name] for name in self.inputs_name]
        targets = [data[name] for name in self.targets_name]

        losses = {}
        acces = {}


        self_recon1, recon1, self_recon2, recon2, e1, c1, s1, e2, c2, s2 = self.cross(inputs[0], inputs[1], targets[0], targets[1])

        #重建损失
        losses['recon1'] = 10*self.l1loss(recon1, targets[0])
        losses['recon2'] = 10*self.l1loss(recon2, targets[1])
        #自我重建损失
        losses['self_recon1'] = self.l1loss(self_recon1, targets[0])
        losses['self_recon2'] = self.l1loss(self_recon2, targets[1])
        # 情感相似性损失
        losses['emo_sim'] = sim_loss(e1, e2)
        # 差异性损失
        losses['diff_1'] = self.diff_loss(e1, c1) + self.diff_loss(e1, s1)
        losses['diff_2'] = self.diff_loss(e2, c2) + self.diff_loss(e2, s2)
        #空间损失
        losses['space'] = self.space_loss(e1, e2 , 2)



        outputs_dict = {
            "self_recon1": self_recon1,
            "recon1": recon1,
            "self_recon2": self_recon2,
            "recon2": recon2,
        }

        return outputs_dict, losses

    def forward(self, x):
        c = self.con_encoder(x)
        e = self.emo_encoder(x)

        d = torch.cat([c, e], dim=1)
        d = self.decoder(d)
        return d

    def update_network(self, loss_dcit):

        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, data):

        self.classify.train()
        self.decoder.train()
        self.spe_encoder.train()
        self.con_encoder.train()
        self.emo_encoder.train()

        outputs, losses, acces = self.process(data)

        self.update_network(losses)

        return outputs, losses, acces

    def val_func(self, data):
        self.classify.eval()
        self.decoder.eval()
        self.con_encoder.eval()
        self.emo_encoder.eval()
        self.spe_encoder.eval()
        with torch.no_grad():
            outputs, losses, acces = self.process(data)

        return outputs, losses, acces

class Wav2vec_con(nn.Module):
    def __init__(self):
        super(Wav2vec_con, self).__init__()
        model_path = "/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/models/chinese-wav2vec2-base"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.wav2Vec = Wav2Vec2Model.from_pretrained(model_path)
        # self.processor = Wav2Vec2Processor.from_pretrained("/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/models/wav2vec2-large-xlsr-53-chinese-zh-cn")
        # self.wav2Vec = Wav2Vec2Model.from_pretrained("/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/models/wav2vec2-base-finetuned-Speaker-Classification")


    def forward(self, x):
        return self.wav2Vec(x).last_hidden_state

    def process(self, x):
        return self.processor(x, sampling_rate=16000, return_tensors="pt").input_values


class Wav2vecNet_con(nn.Module):
    def __init__(self,config):
        super(Wav2vecNet_con, self).__init__()
        self.content_eocder = Wav2vec_con()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, audio):
        x = audio.squeeze(1)
        x = self.content_eocder.process(x)
        x = x.permute(1, 0, 2)
        x = x.squeeze(1)
        x = x.to('cuda')
        x = self.content_eocder(x)
        # x = self.squeeze(self.gap(x))
        # x = self.dropout(x)
        return x

class Wav2vec_spe(nn.Module):
    def __init__(self):
        super(Wav2vec_spe, self).__init__()
        # self.processor = Wav2Vec2Processor.from_pretrained("/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/models/wav2vec2-large-xlsr-53-chinese-zh-cn")
        # self.wav2Vec = Wav2Vec2Model.from_pretrained("/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/models/wav2vec2-base-finetuned-Speaker-Classification")
        model_path = "/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/models/wav2vec2-base-superb-sid"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.wav2Vec = Wav2Vec2Model.from_pretrained(model_path)

    def forward(self, x):
        return self.wav2Vec(x).last_hidden_state

    def process(self, x):
        return self.processor(x, sampling_rate=16000, return_tensors="pt").input_values


class Wav2vecNet_spe(nn.Module):
    def __init__(self,config):
        super(Wav2vecNet_spe, self).__init__()
        self.speaker_eocder = Wav2vec_spe()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, audio):
        x = audio.squeeze(1)
        x = self.speaker_eocder.process(x)
        x = x.permute(1, 0, 2)
        x = x.squeeze(1)
        x = x.to('cuda')
        x = self.speaker_eocder(x)# bt ti 768 *3    ti
        # x = self.squeeze(self.gap(x))# bt 314
        # x = self.dropout(x)
        # 64 322 768
        return x




# class ECAPANet_spe(nn.Module):
#     def __init__(self,config):
#         super(ECAPANet_spe, self).__init__()
#         self.speaker_encoder = EncoderClassifier.from_hparams(source="/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/models/spkrec-ecapa-voxceleb")
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.squeeze = Squeeze(-1)
#         self.dropout = nn.Dropout(config.dropout)
#     def forward(self, audio):
#         x = audio.squeeze(1)
#         x = x.to('cuda')
#         x = self.speaker_eocder.encode_batch(x)
#         x = self.squeeze(self.gap(x))
#         x = self.dropout(x)
#         return x



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(322*3, 256, kernel_size=6, stride=(4,20), padding=1, bias=True),# out_size = (in_size - 1) * S + K - 2P + output_padding
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=(4,2), stride=(4,2), padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=(4,2), padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=(4,3), stride=(4,1), padding=(3,1), bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 1, kernel_size=3, stride=(3,1), padding=1, bias=True),

                nn.Tanh(),
                )

    def forward(self, emotion, content, speaker):
        features = torch.cat([content,  emotion, speaker], 1) #connect tensors inputs and dimension
        features = torch.unsqueeze(features,2)
        features = torch.unsqueeze(features,3)
        x = 90*self.decon(features) #[64, 1, 628, 12] 64 640

        return x

class Classify(nn.Module):
    def __init__(self, dropout=0.25):
        super(Classify, self).__init__()
        # self.last_fc = ResNet(256, 2)
        self.last_fc = nn.Sequential(
            nn.Linear(314, 128),
            # nn.Dropout(dropout),
            # nn.ReLU(True),
            nn.Linear(128, 2),
            # nn.Dropout(dropout),
            # nn.Linear(64, 2),
        )
    def forward(self, feature):
        x = self.last_fc(feature)

        return x

class AudioNet_3Encoder(nn.Module):
    def __init__(self,config):
        super(AudioNet_3Encoder, self).__init__()
        self.args = config
        self.hidden = int(config.audio_length * 50 - 1)
        self.batchsize = config.batch_size
        self.channels = config.output_channels

        self.emo_encoder = E_encoder(self.args, self.channels // 2 ** (config.n_blocks - 1), self.hidden)
        self.con_encoder = Wav2vecNet_con(config)
        self.spe_encoder = Wav2vecNet_spe(config)

        # 冻结 con_encoder 的参数
        for param in self.con_encoder.parameters():
            param.requires_grad = False

        # 冻结 spe_encoder 的参数
        for param in self.spe_encoder.parameters():
            param.requires_grad = False

        # self.decoder = Decoder_pre()
        self.decoder = Decoder_new()
        self.batch_norm = nn.BatchNorm2d(num_features=1)
        self.dropout = nn.Dropout(p=config.dropout)

        self.cosine_loss = nn.CosineSimilarity(dim=-1)
        self.diff_loss = DiffLoss()
        self.CMD_loss = CMD()
        self.tripletloss = nn.TripletMarginLoss(margin=1)
        self.l1loss = nn.L1Loss()
        self.sim_loss = SimLoss()
        self.wav_name = ['wav1', 'wav2']
        self.mel_name = ['mel1', 'mel2']




    def cross(self, audio1, audio2, mfcc1, mfcc2):
        c1 = self.con_encoder(audio1)
        s1 = self.spe_encoder(audio1)
        e1 = self.emo_encoder(mfcc1)

        c2 = self.con_encoder(audio2)
        s2 = self.spe_encoder(audio2)
        e2 = self.emo_encoder(mfcc2)

        self_recon1 = self.decoder(e1, c1, s1)
        recon1 = self.decoder(e2, c1, s1)

        self_recon2 = self.decoder(e2, c2, s2)
        recon2 = self.decoder(e1, c2, s2)

        return self_recon1, recon1, self_recon2, recon2, e1, c1, s1, e2, c2, s2

    def process(self, data):

        wavs = [data[name] for name in self.wav_name]
        mels = [data[name] for name in self.mel_name]
        targets = [mel.detach().clone() for mel in mels]


        losses = {}

        self_recon1, recon1, self_recon2, recon2, e1, c1, s1, e2, c2, s2 = self.cross(wavs[0].cuda(), wavs[1].cuda(), mels[0].cuda(), mels[1].cuda())
        #
        # plot_mel_spectrogram(targets[0][0])
        # plot_mel_spectrogram(targets[0][1])
        #
        # plot_mel_spectrogram(recon1[0])
        # plot_mel_spectrogram(self_recon1[0])

        # #重建后解耦损失
        # recon1_e2 = self.emo_encoder(recon1)
        # recon2_e1 = self.emo_encoder(recon2)

        # losses['recon1_e2'] = 10 * self.tripletloss(recon1_e2.cuda(), e2.cuda(), c2.cuda()) + 10 * self.tripletloss(recon1_e2.cuda(), e2.cuda(), s2.cuda()) + 2*sim_loss(recon1_e2.cuda(), e2)
        # losses['recon2_e1'] = 10 * self.tripletloss(recon2_e1.cuda(), e1.cuda(), c1.cuda()) + 10 * self.tripletloss(recon2_e1.cuda(), e1.cuda(), s1.cuda()) + 2*sim_loss(recon2_e1.cuda(), e1)
        #
        # losses['recon1_e2'] = 10*self.l1loss(recon1_e2.cuda(), e2.cuda())
        # losses['recon2_e1'] = 10*self.l1loss(recon2_e1.cuda(), e1.cuda())

        # losses['recon1_e2'] = 1*sim_loss(recon1_e2.cuda(), e2)
        # losses['recon2_e1'] = 1*sim_loss(recon2_e1.cuda(), e1)

        #重建损失
        # losses['recon1'] = 1*self.l1loss(recon1.cuda(), targets[0].cuda())
        # losses['recon2'] = 1*self.l1loss(recon2.cuda(), targets[1].cuda())

        losses['recon1'] = 500*(1 - self.cosine_loss(recon1.cuda(), targets[0].cuda()).mean())
        losses['recon2'] = 500*(1 - self.cosine_loss(recon2.cuda(), targets[1].cuda()).mean())
        #自我重建损失
        # losses['self_recon1'] = 1*self.l1loss(self_recon1.cuda(), targets[0].cuda())
        # losses['self_recon2'] = 1*self.l1loss(self_recon2.cuda(), targets[1].cuda())

        losses['self_recon1'] = 500*(1 - self.cosine_loss(self_recon1.cuda(), targets[0].cuda()).mean())
        losses['self_recon2'] = 500*(1 - self.cosine_loss(self_recon2.cuda(), targets[1].cuda()).mean())

        # 情感相似性损失
        # losses['emo_sim'] = 5*sim_loss(e1, e2)
        # losses['emo_sim'] = 100 * self.CMD_loss(e1, e2, 3)
        losses['emo_sim'] = 500 * self.sim_loss(e1, e2)

        # # 差异性损失
        losses['diff_1'] = 1000*(self.diff_loss(e1, c1) + self.diff_loss(e1, s1))
        losses['diff_2'] = 1000*(self.diff_loss(e2, c2) + self.diff_loss(e2, s2))

        #空间损失
        x, y, z = e1.shape
        gaussian = torch.randn((x, y, z)).cuda()
        losses['space'] = (self.CMD_loss(e1, gaussian, 3) + self.CMD_loss(e2, gaussian, 3))*0.005

        outputs_dict = {
            "self_recon1": self_recon1,
            "recon1": recon1,
            "self_recon2": self_recon2,
            "recon2": recon2,
        }

        return outputs_dict, losses

    def forward(self, data):
        wavs = [data[name] for name in self.wav_name]
        mels = [data[name] for name in self.mel_name]

        audio1 = wavs[0].cuda()
        audio2 = wavs[1].cuda()
        mfcc1 = mels[0].cuda()
        mfcc2 = mels[1].cuda()
        e1 = self.emo_encoder(mfcc1)
        s1 = self.spe_encoder(audio1)
        c1 = self.con_encoder(audio1)

        e2 = self.emo_encoder(mfcc2)
        s2 = self.spe_encoder(audio2)
        c2 = self.con_encoder(audio2)


        return e1, s1, c1, e2, s2, c2

class Decoder_pre(nn.Module):
    def __init__(self):
        super(Decoder_pre, self).__init__()
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(322*3, 640, kernel_size=(8,3), stride=(8,1), padding=(0,1), bias=True),# out_size = (in_size - 1) * S + K - 2P + output_padding
                nn.BatchNorm2d(640),
                nn.ReLU(True),
                nn.ConvTranspose2d(640, 640, kernel_size=(2,3), stride=(2,1), padding=(0,1), bias=True),
                nn.BatchNorm2d(640),
                nn.ReLU(True),
                nn.ConvTranspose2d(640, 640, kernel_size=(2,3), stride=(2,1), padding=(0,1), bias=True),
                nn.BatchNorm2d(640),
                nn.ReLU(True),
                nn.ConvTranspose2d(640, 640, kernel_size=(2,3), stride=(2,1), padding=(0,1), bias=True),

                nn.Tanh(),
                )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)

    def forward(self, emotion, content, speaker):
        content = self.squeeze(self.gap(content))
        emotion = self.squeeze(self.gap(emotion))
        speaker = self.squeeze(self.gap(speaker))
        features = torch.cat([content,  emotion, speaker], 1) #connect tensors inputs and dimension
        features = torch.unsqueeze(features,2)
        features = torch.unsqueeze(features,3)
        # features = features.permute(0,2,1,3)
        x = self.decon(features) #[bt, 640, 64, 1]
        x = x.squeeze()
        x = x.permute(0, 2, 1)

        return x

class Decoder_new(nn.Module):
    def __init__(self):
        super(Decoder_new, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(768 * 3, 1024, kernel_size=4, stride=2, padding=1),  # 输入通道是拼接后的总通道数
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=1, padding=1),  # 转换成目标形状
        )

    def forward(self, emotion, content, speaker):
        #3个形状都为(322 768) 需要变成(64 640)
        features = torch.cat([content,  emotion, speaker], dim=-1) #322 768*3
        features = features.permute(0, 2, 1)
        x = self.decoder(features)#[bt, 64, 640]

        return x
