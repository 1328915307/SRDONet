import functools

import math
import numpy as np
import torch.nn
from torch import nn
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from adolescent_test.PatchMerging import PatchMerging, CSWinBlock, PatchMerging_noFAA

from adolescent_test.Classifler.models.ResNet import ResNet, ResNet_sup
from adolescent_test.Classifler.models.layers import Squeeze, ConvBlock, Add, BN1d
from models.models_emo_con import Conv2dLSTM_Audio

class ResBlock(nn.Module):
    def __init__(self, ni, n1, n2, nf):
        super(ResBlock, self).__init__()
        self.convblock1 = ConvBlock(ni, n1, kernel_size=3, separable=False, dropout=0.1)
        self.convblock2 = ConvBlock(n1, n2, kernel_size=3, separable=False, dropout=0.1)
        self.convblock3 = ConvBlock(n2, nf, kernel_size=3, act=None, separable=False)
        # expand channels for the sum if necessary
        self.shortcut = BN1d(ni) if ni == nf else ConvBlock(ni, nf, 1, act=None, separable=False)
        self.add = Add()
        self.act = nn.LeakyReLU(0.2)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        temp = self.shortcut(res)
        x = self.add(x, temp)  #pred_CT的第一个block
        x = self.act(x)
        return x

class ConvTransform(nn.Module):
    def __init__(self, cin, tm):
        super(ConvTransform, self).__init__()
        self.conv1 = ConvBlock(cin, 256, kernel_size=3, separable=False, dropout=0.1)
        self.conv2 = ConvBlock(256, 512, kernel_size=3, separable=False, dropout=0.1)
        self.conv3 = ConvBlock(512, 768, kernel_size=3, act=None, separable=False)
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(160, tm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)# (batchsize, 768, 160)
        x = self.act(x)
        # x = x.permute(0, 2, 1)  # (batchsize, 160, 768)
        x = self.fc(x)
        return x
class ConvTransform2(nn.Module):
    def __init__(self, cin, tm):
        super(ConvTransform2, self).__init__()
        self.conv1 = ConvBlock(cin, 256, kernel_size=3, separable=False, dropout=0.1)
        self.conv2 = ConvBlock(256, 512, kernel_size=3, separable=False, dropout=0.1)
        self.conv3 = ConvBlock(512, 768, kernel_size=3, separable=False, dropout=0.1)
        self.conv4 = ConvBlock(160, tm, kernel_size=3, act=None, separable=False)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)# (batchsize, 768, 160)
        x = x.permute(0, 2, 1)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        # x = self.act(x)
        return x




class Audionet_InfoAttentionClassifier(nn.Module):
    def __init__(self,config):
        super(Audionet_InfoAttentionClassifier, self).__init__()


        self.args = config
        self.hidden = int(config.audio_length * 50 - 1)
        self.batchsize = config.batch_size
        self.channels = config.output_channels
        # self.conv_t_f = Stem(1, 16)

        self.emo_encoder = E_encoder(self.args, self.channels // 2 ** (config.n_blocks - 1), self.hidden)
        # self.emo_encoder = Conv2dLSTM_Audio(config)
        if config.classflag == 0:
            self.classify = E_classfier(c_out=2, c_in=self.hidden)
        elif config.classflag ==1:
            self.classify = E_classfier2(c_out=2, c_in=self.hidden)
        self.CroEn_loss =  nn.CrossEntropyLoss(reduction='sum')
        self.l1loss = nn.L1Loss()
        self.dropout = nn.Dropout(p=config.dropout)



    def forward(self, x):

        # x = x.unsqueeze(1)
        # x = x.permute(0, 1, 3, 2)#bt 1 640 64
        # x = self.conv_t_f(x)
        # x = x.squeeze(1)
        # x = x.permute(0, 2, 1)
        output_ = self.emo_encoder(x)#322
        output_ = self.classify(output_)#2
        return output_


    def process(self, data):

        labels = data['label1']
        inputs = data['input1']
        inputs = inputs.cuda()
        # inputs = self.conv_t_f(inputs)
        e1 = self.emo_encoder(inputs)
        #分类损失
        label1 = labels
        # label1 = torch.squeeze(label1)
        one_hot_labels = Variable(torch.zeros(labels.shape[0], 2).scatter_(1, label1.view(-1, 1), 1).cuda())
        one_hot_labels = one_hot_labels.to('cuda')

        pred_label = self.classify(e1)
        loss = self.CroEn_loss(pred_label, one_hot_labels)
        pre1 = torch.argmax(pred_label, dim=1).cpu().numpy()
        label1 = label1.cpu().numpy()
        # acc = self.compute_acc(label1, pre1)
        # f1 = f1_score(label1, pre1)


        return loss, label1, pre1

    def val_func(self, data):
        self.classify.eval()
        self.emo_encoder.eval()
        # self.conv_t_f.eval()

        with torch.no_grad():
            loss, label1, pre1 = self.process(data)

        return loss, label1, pre1

class E_encoder(nn.Module):
    def __init__(self, args, c_in, c_out):
        super(E_encoder, self).__init__()
        nf = 32  # 根据merge后的大小设置通道
        self.args = args
        # if args.flag == 0:
        #     self.conv_t_f = Stem(1, 16)
        # elif args.flag == 1:
        self.conv_t_f = Stem2(1, 16, args.weight)
        self.spectralFusion = AudioInfoCollect2(args)
        self.layernorm = nn.LayerNorm(args.frame_num, elementwise_affine=False)

        self.resblock1 = ResBlock(c_in, 32, 64, 128)
        self.resblock2 = ResBlock(128, 256, 512, 768)
        self.resblock3 = ResBlock(160, 256, 256, 322)

        self.dropout = nn.Dropout(args.dropout)  # 防止over-fitting
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)#bt 1 640 64
        x = self.conv_t_f(x)
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.spectralFusion(x)#16 160
        x = self.dropout(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = x.permute(0, 2, 1)
        x = self.resblock3(x)

        return x
class E_encoder_noTFM(nn.Module):
    def __init__(self, args, c_in, c_out):
        super(E_encoder_noTFM, self).__init__()
        nf = 32  # 根据merge后的大小设置通道

        self.spectralFusion = AudioInfoCollect_noTFM(args)
        self.layernorm = nn.LayerNorm(args.frame_num, elementwise_affine=False)

        self.resblock1 = ResBlock(c_in, 32, 64, 128)
        self.resblock2 = ResBlock(128, 256, 512, 768)
        self.resblock3 = ResBlock(160, 256, 256, 322)
        self.dropout = nn.Dropout(args.dropout)  # 防止over-fitting

    def forward(self, x):
        x = self.layernorm(x)
        x = self.spectralFusion(x)#16 160
        x = self.dropout(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = x.permute(0, 2, 1)
        x = self.resblock3(x)

        return x
class E_encoder_noFAA(nn.Module):
    def __init__(self, args, c_in, c_out):
        super(E_encoder_noFAA, self).__init__()
        nf = 32  # 根据merge后的大小设置通道
        self.args = args
        if args.flag == 0:
            self.conv_t_f = Stem(1, 16)
        elif args.flag == 1:
            self.conv_t_f = Stem2(1, 16, args.weight)
        self.spectralFusion = AudioInfoCollect_noFAA(args)
        self.layernorm = nn.LayerNorm(args.frame_num, elementwise_affine=False)

        self.resblock1 = ResBlock(c_in, 32, 64, 128)
        self.resblock2 = ResBlock(128, 256, 512, 768)
        self.resblock3 = ResBlock(160, 256, 256, 322)

        self.dropout = nn.Dropout(args.dropout)  # 防止over-fitting
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)#bt 1 640 64
        x = self.conv_t_f(x)
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.spectralFusion(x)#16 160
        x = self.dropout(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = x.permute(0, 2, 1)
        x = self.resblock3(x)

        return x
class E_encoder2(nn.Module):
    def __init__(self, args, c_in, c_out):
        super(E_encoder2, self).__init__()
        nf = 32  # 根据merge后的大小设置通道
        self.args = args

        # self.f_b_weight = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.conv_t_f = Stem(1, 16)
        self.spectralFusion = AudioInfoCollect2(args)
        self.layernorm = nn.LayerNorm(args.frame_num, elementwise_affine=False)

        self.resblock1 = ResBlock(c_in, nf)
        self.resblock2 = ResBlock(nf, nf * 2)
        self.resblock3 = ResBlock(nf * 2, nf * 4)
        # self.gap = nn.AdaptiveAvgPool1d(1)
        # self.squeeze = Squeeze(-1)
        # self.fc = nn.Linear(160, c_out)
        self.trans = ConvTransform(nf*4, c_out)
        self.dropout = nn.Dropout(args.dropout)  # 防止over-fitting
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # inputs = melBanks(x, self.args)#64 640

        # x = x.unsqueeze(1)
        # x = x.permute(0, 1, 3, 2)#bt 1 640 64
        # x = self.conv_t_f(x)
        # x = x.squeeze(1)
        # x = x.permute(0, 2, 1)
        # x = self.layernorm(x)
        #
        # forward = x
        # backward = torch.flip(x, [2])
        #
        # forward = self.spectralFusion(forward)#16 160
        # forward = self.resblock1(forward)
        # forward = self.resblock2(forward)
        # forward = self.resblock3(forward)#64 160
        #
        # backward = self.spectralFusion(backward)#16 160
        # backward = self.resblock1(backward)
        # backward = self.resblock2(backward)
        # backward = self.resblock3(backward)#64 160
        #
        # x = (forward*self.f_b_weight) + (backward*(1-self.f_b_weight))

        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)#bt 1 640 64
        x = self.conv_t_f(x)
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = self.layernorm(x)
        x = self.spectralFusion(x)#16 160
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)#128 160(512 256)
        x = self.dropout(x)
        x = self.trans(x)#768 322
        x = x.permute(0, 2, 1)

        # x = self.sigmoid(x)
        return x

class E_classfier(nn.Module):
    def __init__(self, c_out, c_in):
        super(E_classfier, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(322, 128)
        self.fc2 = nn.Linear(128, 2)
        # self.fc3 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):# 322 768
        # x = self.sigmoid(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class E_classfier2(nn.Module):
    def __init__(self, c_out, c_in):
        super(E_classfier2, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(322, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):# 322 768
        # x = self.sigmoid(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


class RegionalExtraction1(nn.Module):
    def __init__(self, args):
        super(RegionalExtraction1, self).__init__()
        self.channels = args.output_channels
        self.spectralFusion = AudioInfoCollect(args)
        # 添加projector
        self.backbone = ResNet_sup(self.channels // 2 ** (args.n_blocks - 1), 2)

    def forward(self, inputs):
        """分类模型"""
        output_ = self.spectralFusion(inputs)
        output_ = self.backbone(output_)
        return output_


class RegionalExtraction2(nn.Module):
    def __init__(self, args):
        super(RegionalExtraction2, self).__init__()
        self.channels = args.output_channels
        self.spectralFusion = AudioInfoCollect(args)
        # 添加projector
        self.backbone = ResNet_sup(self.channels // 2 ** (args.n_blocks - 1), 2)

    def forward(self, inputs):
        """分类模型"""
        output_ = self.spectralFusion(inputs)
        output_ = self.backbone(output_)
        return output_


class AudioInfoCollect2(torch.nn.Module):
    def __init__(self, args):
        """Inititalize variables."""
        super(AudioInfoCollect2, self).__init__()
        self.output_channels = args.output_channels#64
        self.hidden_channels = args.hidden_channels#64
        self.skip_channels = args.skip_channels#64
        self.n_layers = args.n_layers#5
        self.n_blocks = args.n_blocks#3
        self.dilation = args.dilation#2
        # 像素 --------------------
        self.filter_num = args.filter_num
        self.frame_num = args.frame_num
        # 像素 --------------------

        self.dilations = [args.dilation ** i for i in range(args.n_layers)] * args.n_blocks
        self.input_dim = int(args.input_dim)
        self.kernel_size = args.kernel_size#3
        self.relu = nn.LeakyReLU(0.2)
        self.conv, self.skip, self.downsample, self.norm = [], [], [], []
        hidden_channels = self.hidden_channels
        for idx, d in enumerate(self.dilations):
            skip_tmp = nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, bias=False),
                # nn.LayerNorm(self.frame_num // (2 ** (idx // self.n_layers)))
                )
            self.skip.append(skip_tmp)
            # self.resi.append(nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1,
            # bias=False))
            conv_tmp = torch.nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=self.kernel_size,
                          bias=False, dilation=d, padding=d * (self.kernel_size - 1) // 2, groups=hidden_channels),
                # nn.BatchNorm1d(hidden_channels)
                )
            self.conv.append(conv_tmp)
            if (idx + 1) % self.n_layers == 0:  # 如果能被整除
                hidden_channels = self.hidden_channels // (2 ** ((idx + 1) // self.n_layers))
            # ============9==========================
        for block in range(self.n_blocks):
            input_resolution = (self.filter_num // (2 ** block), self.frame_num // (2 ** block))
            self.downsample.append(PatchMerging(input_resolution, dim=16, block=block))
            # self.downsample.append(CSWinBlock( dim = 16, reso = input_resolution, num_heads = 4, split_size = 4))
            # self.norm.append(nn.LayerNorm(self.frame_num // (2 ** block)))
        self.conv = nn.ModuleList(self.conv)
        self.skip = nn.ModuleList(self.skip)
        self.downsample = nn.ModuleList(self.downsample)


    def forward(self, inputs: torch.Tensor):
        """Returns embedding vectors. """
        output = inputs
        skip_connections = []
        for idx, (dilation, conv, skip) in enumerate(zip(self.dilations, self.conv, self.skip)):
            if output.dim() == 2:
                # 在最前面添加一个大小为 1 的维度
                output = output.unsqueeze(0)
            shortcut = output
            "dilated"
            output = conv(output)
            "skip-connectiuon"
            output = self.relu(output)
            skip_outputs = skip(output)
            skip_connections.append(skip_outputs)
            # output = resi(output) # 去掉，降低网络层数
            "resNet"
            output = output + shortcut[:, :, -output.size(2):]
            if dilation == 2 ** (self.n_layers - 1):  # 每个block单独进行特征汇聚
                # 定义一个权重,来动态融合每个block中的多尺度特征, 每个block都提取了不同尺度的信息. 但是融合的时候要有策略.，这个权重就是skip_tmp中卷积核为1的一维卷积
                sum_output = sum([s[:, :, -output.size(2):] for s in skip_connections])
                # output = self.norm[((idx + 1) // self.n_layers)-1](output)
                """
                要充分挖掘时间T 和 频率 F 上的信息  
                1. 多尺度汇聚
                2. 降低维度， 在每个block中， TCN中时间序列的维度并未改变。B F T
                # PatchMerging的优势：可以完整保留相应区域中的信息。
                """
                if ((idx + 1) // self.n_layers) <= self.n_blocks - 1:
                    output = self.downsample[((idx + 1) // self.n_layers) - 1](sum_output)
                # 清空每个block的多尺度信息, 每次清空
                skip_connections = []

        return output
class AudioInfoCollect_noTFM(torch.nn.Module):
    def __init__(self, args):
        """Inititalize variables."""
        super(AudioInfoCollect_noTFM, self).__init__()
        self.output_channels = args.output_channels#64
        self.hidden_channels = args.hidden_channels#64
        self.skip_channels = args.skip_channels#64
        self.n_layers = args.n_layers#5
        self.n_blocks = args.n_blocks#3
        self.dilation = args.dilation#2
        # 像素 --------------------
        self.filter_num = args.filter_num
        self.frame_num = args.frame_num
        # 像素 --------------------

        self.dilations = [args.dilation ** i for i in range(args.n_layers)] * args.n_blocks
        self.input_dim = int(args.input_dim)
        self.kernel_size = args.kernel_size#3
        self.relu = nn.LeakyReLU(0.2)
        self.conv, self.skip, self.downsample, self.norm = [], [], [], []
        hidden_channels = self.hidden_channels
        for idx, d in enumerate(self.dilations):
            skip_tmp = nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, bias=False),
                # nn.LayerNorm(self.frame_num // (2 ** (idx // self.n_layers)))
                )
            self.skip.append(skip_tmp)
            # self.resi.append(nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1,
            # bias=False))
            conv_tmp = torch.nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=self.kernel_size,
                          bias=False, dilation=d, padding=d * (self.kernel_size - 1) // 2, groups=hidden_channels),
                # nn.BatchNorm1d(hidden_channels)
                )
            self.conv.append(conv_tmp)
            if (idx + 1) % self.n_layers == 0:  # 如果能被整除
                hidden_channels = self.hidden_channels // (2 ** ((idx + 1) // self.n_layers))
            # ============9==========================
        for block in range(self.n_blocks):
            input_resolution = (self.filter_num // (2 ** block), self.frame_num // (2 ** block))
            self.downsample.append(PatchMerging(input_resolution, dim=16, block=block))
            # self.downsample.append(CSWinBlock( dim = 16, reso = input_resolution, num_heads = 4, split_size = 4))
            # self.norm.append(nn.LayerNorm(self.frame_num // (2 ** block)))
        self.conv = nn.ModuleList(self.conv)
        self.skip = nn.ModuleList(self.skip)
        self.downsample = nn.ModuleList(self.downsample)

    def forward(self, inputs: torch.Tensor):
        output = inputs

        for idx, (dilation, conv, skip) in enumerate(zip(self.dilations, self.conv, self.skip)):
            if output.dim() == 2:
                # 在最前面添加一个大小为 1 的维度
                output = output.unsqueeze(0)
            if dilation == 2 ** (self.n_layers - 1):  # 每个block单独进行特征汇聚
                # 定义一个权重,来动态融合每个block中的多尺度特征, 每个block都提取了不同尺度的信息. 但是融合的时候要有策略.，这个权重就是skip_tmp中卷积核为1的一维卷积
                sum_output = conv(output)
                output = self.relu(output)
                if ((idx + 1) // self.n_layers) <= self.n_blocks - 1:
                    output = self.downsample[((idx + 1) // self.n_layers) - 1](sum_output)

        return output

class AudioInfoCollect_noFAA(torch.nn.Module):
    def __init__(self, args):
        """Inititalize variables."""
        super(AudioInfoCollect_noFAA, self).__init__()
        self.output_channels = args.output_channels#64
        self.hidden_channels = args.hidden_channels#64
        self.skip_channels = args.skip_channels#64
        self.n_layers = args.n_layers#5
        self.n_blocks = args.n_blocks#3
        self.dilation = args.dilation#2
        # 像素 --------------------
        self.filter_num = args.filter_num
        self.frame_num = args.frame_num
        # 像素 --------------------

        self.dilations = [args.dilation ** i for i in range(args.n_layers)] * args.n_blocks
        self.input_dim = int(args.input_dim)
        self.kernel_size = args.kernel_size#3
        self.relu = nn.LeakyReLU(0.2)
        self.conv, self.skip, self.downsample, self.norm = [], [], [], []
        hidden_channels = self.hidden_channels
        for idx, d in enumerate(self.dilations):
            skip_tmp = nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, bias=False),
                # nn.LayerNorm(self.frame_num // (2 ** (idx // self.n_layers)))
                )
            self.skip.append(skip_tmp)
            # self.resi.append(nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1,
            # bias=False))
            conv_tmp = torch.nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=self.kernel_size,
                          bias=False, dilation=d, padding=d * (self.kernel_size - 1) // 2, groups=hidden_channels),
                # nn.BatchNorm1d(hidden_channels)
                )
            self.conv.append(conv_tmp)
            if (idx + 1) % self.n_layers == 0:  # 如果能被整除
                hidden_channels = self.hidden_channels // (2 ** ((idx + 1) // self.n_layers))
            # ============9==========================
        for block in range(self.n_blocks):
            input_resolution = (self.filter_num // (2 ** block), self.frame_num // (2 ** block))
            self.downsample.append(PatchMerging_noFAA(input_resolution, dim=16, block=block))
            # self.downsample.append(CSWinBlock( dim = 16, reso = input_resolution, num_heads = 4, split_size = 4))
            # self.norm.append(nn.LayerNorm(self.frame_num // (2 ** block)))
        self.conv = nn.ModuleList(self.conv)
        self.skip = nn.ModuleList(self.skip)
        self.downsample = nn.ModuleList(self.downsample)


    def forward(self, inputs: torch.Tensor):
        """Returns embedding vectors. """
        output = inputs
        skip_connections = []
        for idx, (dilation, conv, skip) in enumerate(zip(self.dilations, self.conv, self.skip)):
            if output.dim() == 2:
                # 在最前面添加一个大小为 1 的维度
                output = output.unsqueeze(0)
            shortcut = output
            "dilated"
            output = conv(output)
            "skip-connectiuon"
            output = self.relu(output)
            skip_outputs = skip(output)
            skip_connections.append(skip_outputs)
            # output = resi(output) # 去掉，降低网络层数
            "resNet"
            output = output + shortcut[:, :, -output.size(2):]
            if dilation == 2 ** (self.n_layers - 1):  # 每个block单独进行特征汇聚
                # 定义一个权重,来动态融合每个block中的多尺度特征, 每个block都提取了不同尺度的信息. 但是融合的时候要有策略.，这个权重就是skip_tmp中卷积核为1的一维卷积
                sum_output = sum([s[:, :, -output.size(2):] for s in skip_connections])
                if ((idx + 1) // self.n_layers) <= self.n_blocks - 1:
                    output = self.downsample[((idx + 1) // self.n_layers) - 1](sum_output)
                # 清空每个block的多尺度信息, 每次清空
                skip_connections = []

        return output


class AudioInfoCollect(torch.nn.Module):
    def __init__(self, args):
        """Inititalize variables."""
        super(AudioInfoCollect, self).__init__()
        self.output_channels = args.output_channels  # 64
        self.hidden_channels = args.hidden_channels  # 64
        self.skip_channels = args.skip_channels  # 64
        self.n_layers = args.n_layers  # 5
        self.n_blocks = args.n_blocks  # 3
        self.dilation = args.dilation  # 2
        # 像素 --------------------
        self.filter_num = args.filter_num
        self.frame_num = args.frame_num
        # 像素 --------------------

        self.dilations = [args.dilation ** i for i in range(args.n_layers)] * args.n_blocks
        self.input_dim = int(args.input_dim)
        self.kernel_size = args.kernel_size  # 3
        self.relu = nn.LeakyReLU(0.2)
        self.conv, self.skip, self.downsample, self.norm = [], [], [], []
        hidden_channels = self.hidden_channels
        for idx, d in enumerate(self.dilations):
            skip_tmp = nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, bias=False),
                # nn.LayerNorm(self.frame_num // (2 ** (idx // self.n_layers)))
            )
            self.skip.append(skip_tmp)
            # self.resi.append(nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1,
            # bias=False))
            conv_tmp = torch.nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=self.kernel_size,
                          bias=False, dilation=d, padding=d * (self.kernel_size - 1) // 2, groups=hidden_channels),
                # nn.BatchNorm1d(hidden_channels)
            )
            self.conv.append(conv_tmp)
            if (idx + 1) % self.n_layers == 0:  # 如果能被整除
                hidden_channels = self.hidden_channels // (2 ** ((idx + 1) // self.n_layers))
            # ============9==========================
        for block in range(self.n_blocks):
            input_resolution = (self.filter_num // (2 ** block), self.frame_num // (2 ** block))
            self.downsample.append(PatchMerging(input_resolution, dim=16, block=block))
            # self.downsample.append(CSWinBlock( dim = 16, reso = input_resolution, num_heads = 4, split_size = 4))
            # self.norm.append(nn.LayerNorm(self.frame_num // (2 ** block)))
        self.conv = nn.ModuleList(self.conv)
        self.skip = nn.ModuleList(self.skip)
        self.downsample = nn.ModuleList(self.downsample)

    def forward(self, inputs):
        output = inputs
        skip_connections = []

        for idx, (dilation, conv, skip) in enumerate(zip(self.dilations, self.conv, self.skip)):
            if idx < self.n_layers:
                if output.dim() == 2:
                    # 在最前面添加一个大小为 1 的维度
                    output = output.unsqueeze(0)
                shortcut = output
                "dilated"
                output = conv(output)
                "skip-connectiuon"
                output = self.relu(output)
                skip_outputs = skip(output)
                skip_connections.append(skip_outputs)
                # output = resi(output) # 去掉，降低网络层数
                "resNet"
                output = output + shortcut[:, :, -output.size(2):]
                if dilation == (self.n_layers - 1):
                    output = sum([s[:, :, -output.size(2):] for s in skip_connections])

            if dilation == 2 ** (self.n_layers - 1):  # 每个block单独进行特征汇聚
                if ((idx + 1) // self.n_layers) <= self.n_blocks - 1:
                    output = self.downsample[((idx + 1) // self.n_layers) - 1](output)


        return output


def melBanks(pow_frames,args):
    sample_rate = args.fs
    NFFT = args.NFFT
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    nfilt = args.filter_num
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
    fbank = torch.from_numpy(fbank).to(pow_frames.device)

    pow_frames = pow_frames.double()
    fbank = fbank.double()

    # 执行矩阵乘法，并将结果转换回float类型
    filter_banks = torch.matmul(pow_frames.permute(0, 2, 1), fbank.permute(1, 0)).permute(0, 2, 1).float()
    filter_banks = torch.where(filter_banks == 0, torch.finfo(torch.float).eps, filter_banks)

    filter_banks = 20 * torch.log10(filter_banks)

    return filter_banks

class Stem(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int
    ):
        super().__init__()


        self.t_f_weight = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)


        # self.conv1_t = Conv2dSame(in_channels=in_chs, out_channels=out_chs, kernel_size=(6, 3), stride=(2, 2), groups=1, bias=False)
        # self.conv1_f = Conv2dSame(in_channels=in_chs, out_channels=out_chs, kernel_size=(3, 6), stride=(2, 2), groups=1, bias=False)
        self.conv1_t = Conv2dSame(in_channels=in_chs, out_channels=out_chs, kernel_size=(6, 3), stride=(1, 1), groups=1, bias=False)
        self.conv1_f = Conv2dSame(in_channels=in_chs, out_channels=out_chs, kernel_size=(3, 6), stride=(1, 1), groups=1, bias=False)


        self.norm1_t = nn.BatchNorm2d(out_chs)
        self.norm1_f = nn.BatchNorm2d(out_chs)

        self.act1_t = nn.ReLU()
        self.act1_f = nn.ReLU()


        self.conv2_t = torch.nn.Conv2d(in_channels=out_chs, out_channels=in_chs, kernel_size=(6 ,3), padding='same', stride=1, groups=1, bias=False)
        self.conv2_f = torch.nn.Conv2d(in_channels=out_chs, out_channels=in_chs, kernel_size=(3 ,6), padding='same', stride=1, groups=1, bias=False)


    def forward(self, x):# 640 32

        x_t = self.conv1_t(x)
        x_t = self.norm1_t(x_t)
        x_t = self.act1_t(x_t)
        x_t = self.conv2_t(x_t)

        x_f = self.conv1_f(x)
        x_f = self.norm1_f(x_f)
        x_f = self.act1_f(x_f)
        x_f = self.conv2_f(x_f)

        # x_t = self.conv2_t(self.norm1_t(self.conv1_t(x)))
        # x_f = self.conv2_f(self.norm1_f(self.conv1_f(x)))

        return (x_t*self.t_f_weight) + (x_f*(1-self.t_f_weight))


class Stem2(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            weight: float
    ):
        super().__init__()


        self.t_f_weight = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weight = weight

        # self.conv1_t = Conv2dSame(in_channels=in_chs, out_channels=out_chs, kernel_size=(6, 3), stride=(2, 2), groups=1, bias=False)
        # self.conv1_f = Conv2dSame(in_channels=in_chs, out_channels=out_chs, kernel_size=(3, 6), stride=(2, 2), groups=1, bias=False)
        self.conv1_t = Conv2dSame(in_channels=in_chs, out_channels=out_chs, kernel_size=(6, 3), stride=(1, 1), groups=1, bias=False)
        self.conv1_f = Conv2dSame(in_channels=in_chs, out_channels=out_chs, kernel_size=(3, 6), stride=(1, 1), groups=1, bias=False)
        self.conv1 = Conv2dSame(in_channels=in_chs, out_channels=out_chs, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)


        self.norm1_t = nn.BatchNorm2d(out_chs)
        self.norm1_f = nn.BatchNorm2d(out_chs)
        self.norm1 = nn.BatchNorm2d(out_chs)

        self.act1_t = nn.ReLU()
        self.act1_f = nn.ReLU()
        self.act1 = nn.ReLU()

        self.conv2_t = torch.nn.Conv2d(in_channels=out_chs, out_channels=in_chs, kernel_size=(6 ,3), padding='same', stride=1, groups=1, bias=False)
        self.conv2_f = torch.nn.Conv2d(in_channels=out_chs, out_channels=in_chs, kernel_size=(3 ,6), padding='same', stride=1, groups=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=out_chs, out_channels=in_chs, kernel_size=(3, 3), padding='same', stride=1, groups=1, bias=False)

        self.final_output = nn.Conv2d(in_chs, in_chs, kernel_size=1,bias=False)


    def forward(self, x):# 640 32

        x_t = x
        x_f = x

        x_t = self.conv1_t(x_t)
        x_t = self.norm1_t(x_t)
        x_t = self.act1_t(x_t)
        x_t = self.conv2_t(x_t)

        x_f = self.conv1_f(x_f)
        x_f = self.norm1_f(x_f)
        x_f = self.act1_f(x_f)
        x_f = self.conv2_f(x_f)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)

        # x_t = self.conv2_t(self.norm1_t(self.conv1_t(x)))
        # x_f = self.conv2_f(self.norm1_f(self.conv1_f(x)))
        out = ((x_t * self.t_f_weight) + (x_f * (1 - self.t_f_weight))) * self.weight + x * (1 - self.weight)
        # 使用额外的层来捕获最终输出
        final_out = self.final_output(out)
        return final_out

class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        # print("kernel, ",self.weight.shape)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


