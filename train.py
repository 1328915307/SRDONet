import copy
import json
import os
import pickle
import random
import time
import datetime
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
import numpy as np
import torch
import torch.utils
import torch.nn as nn
from nvidia import cudnn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from collections import OrderedDict
import argparse
from adolescent_test.model_main import  Audionet_InfoAttentionClassifier
from tqdm import tqdm

from adolescent_test.untils import total_dataloader2
from dataset.dataload import MyDataset, MyDatasetone, MyDataset_audio
from torch.nn import init
from dataset.dataload import total_dataloader
from method import evaluate, save
from torchcam.methods import GradCAM

from models.models_emo_con import AutoEncoder2x, Audionet, Audionet_wav2vec


def save_activation_map(activation_map, save_path):
    """
    保存激活映射（注意力热图）为图片文件。

    参数:
        activation_map: 激活映射（numpy array）
        save_path: 保存路径
    """
    # 归一化处理
    activation_map_normalized = normalize(activation_map)

    # 使用matplotlib进行绘制
    plt.figure(figsize=(10, 4))
    plt.imshow(activation_map_normalized, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar()
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 保存梅尔谱图
def save_mel_spectrogram(mel_tensor, path, title=None):
    mel = mel_tensor.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    if title:
        plt.title(title)
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def normalize(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-8)


def overlay_gradcam_on_mel(input_tensor, activation_map, save_path, cmap='rainbow'):
    # 处理梅尔频谱图
    mel = input_tensor.squeeze().detach().cpu().numpy()
    mel_normalized = normalize(mel)

    # 这里不使用 PIL 直接创建图像，而是用 matplotlib 保证和 save_mel_spectrogram 一致
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(mel_normalized, cmap='viridis', aspect='auto', origin='lower')
    ax.axis('off')
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    mel_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)

    mel_image = mel_image.convert("RGBA")

    # 处理激活映射(注意力热图)
    heatmap = normalize(activation_map.cpu().numpy())
    heatmap = cv2.resize(heatmap, (mel.shape[1], mel.shape[0]))
    if cmap == 'gray':
        heatmap_colored = plt.cm.gray(heatmap)
    elif cmap == 'rainbow':
        heatmap_colored = plt.cm.rainbow(heatmap)
    else:
        heatmap_colored = plt.cm.viridis(heatmap)

    heatmap_image = Image.fromarray((heatmap_colored[:, :, :3] * 255).astype(np.uint8))
    heatmap_image = heatmap_image.convert("RGBA")

    # 调整热力图的透明度，让其更突出
    alpha = 0.7
    heatmap_image.putalpha(int(255 * alpha))

    # 确保 heatmap_image 的尺寸和 mel_image 一致
    heatmap_image = heatmap_image.resize(mel_image.size, Image.ANTIALIAS)

    # 创建一个空白的 RGBA 图像，用于叠加
    overlay = Image.new('RGBA', mel_image.size, (0, 0, 0, 0))
    overlay = Image.alpha_composite(overlay, mel_image)
    overlay = Image.alpha_composite(overlay, heatmap_image)

    # 保存最终结果
    overlay.save(save_path)


def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def initialize_weights( net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llr",
                        type=float,
                        default=0.005)
    parser.add_argument("--hlr",
                        type=float,
                        default=0.01)
    parser.add_argument("--lr",
                        type=float,
                        default=0.001)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.9)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--batch_size",
                        type=int,
                        default=64)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=60)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/home/idal-01/code/TD-learnableFilters/CMDC.pkl")#抑郁depressed 焦虑anxious 压力pressure 失眠insomnia
                                #/home/idal-01/code/TD-learnableFilters/CMDC.pkl
                                #/home/idal-01/caoyuhao/data/data_preprosess_new/两次低分抑郁重症匹配.pkl
                                #/home/idal-01/code/TD-learnableFilters/NRAC_L.pkl
                                #/home/idal-01/caoyuhao/data/data_preprosess_new/两次低分抑郁全部匹配.pkl
                                #/home/idal-01/caoyuhao/data/data_preprosess_new/anxious_audio_0123.pkl
                                #/home/idal-01/caoyuhao/data/data_preprosess_new/两次低分压力全部匹配.pkl
    parser.add_argument("--model_dir",
                        type=str,
                        default="/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/save_model/")
    parser.add_argument("--image_dir",
                        type=str,
                        default="/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/image/")
    parser.add_argument("--log_dir",
                        type=str,
                        default="/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/log/")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classflag", type=int, default=0)#0两层1一层
    parser.add_argument("--flag", type=int, default=1)#1有w
    parser.add_argument("--weight", type=float, default=0.7)#本体占1-w
    parser.add_argument("--KFold", type=int, default=1)
    parser.add_argument("--step_size", type=float, default=60)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str, default='')
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--pretrained_dir', type=str,
                        default='/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/save_model/epoch50_CMDC_ecs+convtf+newloss_202406212058.pth')
    #/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/save_model/epoch50_CMDC_ecs+convtf+newloss_202406212058.pth
    #/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/save_model/epoch50_压力_ecs+convtf+newloss_202409121621.pth
    #/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/save_model/epoch50_depress0123_ecs+convtf+oldloss_202406210158.pth
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--rnn', type=bool, default=True)

##################################
    fs = 16000
    # cw_len = 10.3
    cw_len = 6.46
    wlen = int(fs * cw_len)
    parser.add_argument('--input_dim', type=int, default=wlen)
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--audio_length', type=int, default=cw_len)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--output_channels', type=int, default=64)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--skip_channels', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--dilation', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--aptim', type=str, default='adam')
    parser.add_argument('--initializer', type=str, default='random')  # mel_scale
    parser.add_argument('--experiment', type=str, default='adolescent30-60')
    # TD滤波器
    parser.add_argument('--filter_size', type=int, default=513)
    parser.add_argument('--filter_num', type=int, default=64)
    parser.add_argument('--frame_num', type=int, default=640)
    parser.add_argument('--NFFT', type=int, default=1024)
    parser.add_argument('--sigma_coff', type=float, default=0.0015)
##################################

    return parser.parse_args()


def init_seeds(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


init_seeds(1)
device_index = 0  # 目标GPU的索引
torch.cuda.set_device(device_index)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = parse_args()
os.makedirs(config.model_dir,exist_ok = True)
os.makedirs(config.image_dir,exist_ok = True)
os.makedirs(config.log_dir,exist_ok = True)


###########load data#######################
print('start load data')
# # 加载mfcc特征
# train_set = MyDatasetone(config.dataset_dir, mode='train', train_ratio=0.8)
# test_set = MyDatasetone(config.dataset_dir, mode='test', train_ratio=0.8)
# # 加载语音
# train_set = MyDataset_audio(config.dataset_dir, mode='train', train_ratio=0.8)
# test_set = MyDataset_audio(config.dataset_dir, mode='test', train_ratio=0.8)
#
#
# train_loader = DataLoader(train_set,batch_size=config.batch_size,
#                                       num_workers=config.num_thread,
#                                       shuffle=True, drop_last=True)
# test_loader = DataLoader(test_set,batch_size=config.batch_size,
#                                       num_workers=config.num_thread,
#                                       shuffle=True, drop_last=True)
#加载mfcc特征
# train_loaders, test_loaders, val_loaders, toPersonList = total_dataloader(config)

####################代码测试#######################

f = open(config.dataset_dir, 'rb')

data, Labels = pickle.load(f)
# data = [tensor.squeeze().numpy() for tensor in data]

print('Load data finished' + '\n')
print('The number of patient: ', len(np.where(np.array(Labels) == 1)[0]))
print('The number of NC: ', len(np.where(np.array(Labels) == 0)[0]))

train_loaders, test_loaders, val_loaders, toPersonList, ori_test_dataList = total_dataloader2(config, data, Labels)
###########################################



###########train#######################
print('start train')

start_epoch = config.start_epoch


a = time.time()

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
log_path = os.path.join(config.log_dir, f"train_{current_time}.txt")

config_dict = vars(config)  # 将 Namespace 对象转换为字典
# 将配置参数转换为 JSON 格式
config_json = json.dumps(config_dict)
# 将 JSON 字符串写入文件
with open(log_path, 'a') as file_handle:
    file_handle.write(config_json)

best_all_acc = 0
best_all_f1  = 0
results = []
bepoch = []
for train_idx in range(config.KFold):

    best_acc = 0
    best_epoch_acc = 0
    best_f1 = 0
    best_epoch_f1 = 0
    train_iter = 0
    val_iter = 0
    train_loader = train_loaders[train_idx]
    val_loader = val_loaders[train_idx]
    test_loader = test_loaders[train_idx]

    # model = Audionet(config)
    # model = Audionet_wav2vec(config)
    model = Audionet_InfoAttentionClassifier(config).cuda()
    # for name, param in model.named_parameters():
    #     print(name, param.nelement())
    total_info = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: %.6f' % total_info)
    if config.pretrain :
        pretrain = torch.load(config.pretrained_dir)
        model.emo_encoder.load_state_dict(pretrain['emo_encoder'])
        print("Loaded pre-trained parameters into emo_encoder.")
        # tgt_state = model.state_dict()
        # # strip = '0'
        # strip = 'emo_encoder'
        # for name, param in pretrain['emo_encoder'].items():
        #     name = strip+"." + name
        #     if name not in tgt_state:
        #         continue
        #     if isinstance(param, nn.Parameter):
        #         param = param.data
        #     if name in tgt_state:
        #         tgt_state[name].copy_(param)
        #         print(name)
    # # 比较模型参数与预训练参数
    # for name, param in model.emo_encoder.named_parameters():
    #     pretrain_param = pretrain['emo_encoder'][name]
    #     if not torch.allclose(param.data, pretrain_param.data):
    #         print(f"Parameter {name} does not match.")
    #     else:
    #         print(f"Parameter {name} matches.")


    best_acc_model = None
    best_f1_model = None
    last_model = None
    max_w = 0.0
    min_w = 1.0
    criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
    if config.pretrain:
        # 定义参数组
        param_groups = []

        # emo_encoder 模块的参数组
        param_groups.append({
            'params': model.emo_encoder.parameters(),
            'lr': config.llr,
            'weight_decay': 0.001,
        })

        # 其他部分的参数组
        param_groups.append({
            'params': [param for name, param in model.named_parameters() if
                       "emo_encoder" not in name and param.requires_grad],
            'lr': config.hlr,
            'weight_decay': 0.001,
        })

        # 创建优化器
        optimizer1 = torch.optim.Adam(param_groups)
    else:
        optimizer1 = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=config.lr,weight_decay=0.001)
    optim_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=60, gamma=0.95)
    for epoch in range(start_epoch, config.max_epochs):

        epoch_start_time = time.time()
        label_all = []
        pre_all = []
        label_v_all = []
        pre_v_all = []
        total_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()

            inputs = data['input1']
            labels = data['label1']
            pred = model(inputs.cuda())
            label1 = labels
            label1 = torch.squeeze(label1)
            one_hot_labels = Variable(torch.zeros(labels.shape[0], 2).scatter_(1, label1.view(-1, 1), 1).cuda())
            one_hot_labels = one_hot_labels.to('cuda')
            loss = criterion(pred, one_hot_labels)
            model.zero_grad()
            loss.backward()
            optimizer1.step()
            optim_scheduler1.step()


            pre_label = torch.argmax(pred, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            total_loss +=loss
            pre_all.append(pre_label)
            label_all.append(labels)

            total_batches = len(train_loader)

            # if (train_iter % (total_batches // 4) == 0):
            #     # print('epoch:{} [{}/{}]    train_loss: {:.10f}'.format(epoch + 1, i + 1, len(train_loader), loss))
            #     with open(log_path, 'a') as file_handle:
            #         file_handle.write('epoch:{} [{}/{}]    train_loss: {:.10f}'.format(epoch + 1, i + 1, len(train_loader), loss))
            train_iter += 1
        print('epoch:{} total_train_loss: {:.10f}'.format(epoch + 1,total_loss / len(train_loader)))

        pre_all = np.concatenate(pre_all)
        label_all = np.concatenate(label_all)
        acc = accuracy_score(label_all, pre_all)
        f1 = f1_score(label_all, pre_all)
        print(f'train_Acc: {acc} train_f1: {f1}\n')
        with open(log_path, 'a') as file_handle:
            file_handle.write(f'train_Acc: {acc} train_f1: {f1}\n')
        # 计算混淆矩阵
        [[TN, FP], [FN, TP]]= confusion_matrix(label_all, pre_all)



        confusion_matrix_info = (
            f'Train Confusion Matrix:\n'
            f'-----------------------------\n'
            f'| TP: {TP:3} | FN: {FN:3} |\n'
            f'-----------------------------\n'
            f'| FP: {FP:3} | TN: {TN:3} |\n'
            f'-----------------------------\n'
        )

        # 打印到控制台
        print(confusion_matrix_info)

        print("start to validate, epoch %d:" %(epoch+1))

        # with torch.no_grad():
        #     for i, data in enumerate(test_loader):
        #         inputs = data['input1']
        #         labels = data['label1']
        #         pred = model(inputs.cuda())
        #         label1 = labels
        #         label1 = torch.squeeze(label1)
        #         pre_label = torch.argmax(pred, dim=1).cpu().numpy()
        #         labels = labels.cpu().numpy()
        #         pre_v_all.append(pre_label)
        #         label_v_all.append(labels)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                loss, label, pre = model.val_func(data)
                pre_v_all.append(pre)
                label_v_all.append(label)

        # print("--------------开始绘制CAM注意力热图---------------------")
        # model.train()  # 设置为评估模式
        # # 创建保存Grad-CAM图像的目录
        # save_dir = os.path.join("/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/CAM",
        #                         f"epoch_{epoch + 1}_CMDC_{current_time}")
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # # 假设你想用前n个测试数据作为例子
        # n_samples = 5  # 可以根据需要调整
        # sample_idx = 0  # 记录处理了多少样本
        #
        # for i, data in enumerate(test_loader):
        #     if sample_idx >= n_samples:
        #         break
        #
        #     inputs = data['input1'].cuda()
        #     labels = data['label1']
        #
        #     for j in range(min(n_samples - sample_idx, inputs.size(0))):
        #         input_j = inputs[j:j + 1]  # shape: (1, C, H, W)
        #         input_j = input_j.clone().detach().requires_grad_(True)
        #
        #         # Forward pass 获取预测结果
        #
        #         output_j = model(input_j)
        #         pred_label = output_j.argmax().item()
        #         # with torch.no_grad():
        #         #     features = model.emo_encoder.conv_t_f.final_output(input_j)
        #         #     print("Feature map mean:", features.mean().item())  # 应为非零
        #         # 初始化Grad-CAM方法
        #         cam_extractor = GradCAM(model, target_layer=model.emo_encoder.conv_t_f.final_output)
        #         # 使用原始输入图像进行 Grad-CAM 可视化
        #         activation_map = cam_extractor(
        #             class_idx=output_j.squeeze(0).argmax().item(),
        #             scores=output_j,
        #         )
        #
        #         # 保存 Mel 图和热力图
        #         mel_save_path = os.path.join(save_dir, f"sample_{sample_idx + 1}_mel.png")
        #         save_mel_spectrogram(input_j[0], mel_save_path, title=f"Sample {sample_idx + 1}")
        #
        #         # 单独保存热力图
        #         activation_map_save_path = os.path.join(save_dir, f"sample_{sample_idx + 1}_activation_map.png")
        #         save_activation_map(activation_map[0].squeeze().cpu().numpy(), activation_map_save_path)
        #
        #         overlay_gradcam_on_mel(
        #             input_j[0],
        #             activation_map[0].squeeze(),
        #             os.path.join(save_dir, f"sample_{sample_idx + 1}_gradcam_overlay.png")
        #         )
        #         cam_extractor.remove_hooks()
        #         sample_idx += 1




        pre_v_all = np.concatenate(pre_v_all)
        label_v_all = np.concatenate(label_v_all)
        acc_v = accuracy_score(label_v_all, pre_v_all)
        f1_v = f1_score(label_v_all, pre_v_all)
        # 计算混淆矩阵
        cm = confusion_matrix(label_v_all, pre_v_all)

        # 打印混淆矩阵的四个值
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]


        print(f'val_Acc: {acc_v}  val_f1: {f1_v}\n')
        with open(log_path, 'a') as file_handle:
            file_handle.write(f'val_Acc: {acc_v}  val_f1: {f1_v}\n')

        # 格式化混淆矩阵信息
        confusion_matrix_info = (
            f'val Confusion Matrix:\n'
            f'-----------------------------\n'
            f'| TP: {TP:3} | FN: {FN:3} |\n'
            f'-----------------------------\n'
            f'| FP: {FP:3} | TN: {TN:3} |\n'
            f'-----------------------------\n'
        )

        # 打印到控制台
        print(confusion_matrix_info)
        weight = model.emo_encoder.conv_t_f.t_f_weight
        w = weight.item()
        if w < min_w:
            min_w = w
        if w > max_w:
            max_w = w
        print(f"时频卷积模块w的值： {weight.data}")

        # 写入文件
        with open(log_path, 'a') as file_handle:
            file_handle.write(confusion_matrix_info)



        if epoch >= 1:
            if (acc_v >= best_acc):
                best_epoch_acc = epoch+1
                best_acc = acc_v
                best_acc_model = copy.deepcopy(model)
            if (f1_v >= best_f1):
                best_epoch_f1 = epoch+1
                best_f1 = f1_v
                best_f1_model = copy.deepcopy(model)
    last_model = model
    print(f'best_Acc: {best_acc} in epoch:{best_epoch_acc}\n')
    print(f'best_f1: {best_f1} in epoch:{best_epoch_f1}\n')
    with open(log_path, 'a') as file_handle:
        file_handle.write(f'best_Acc: {best_acc} in epoch:{best_epoch_acc}   best_f1: {best_f1} in epoch:{best_epoch_f1}\n')
    best_all_acc = max(best_all_acc, best_acc)
    best_all_f1 = max(best_all_f1, best_f1)

    print('Testing..........' + '\n')
    result_acc = evaluate(test_loader, best_acc_model, toPersonList[train_idx])
    result_f1 = evaluate(test_loader, best_f1_model, toPersonList[train_idx])
    result_last = evaluate(test_loader, last_model, toPersonList[train_idx])

    f1 = result_acc['acc']
    f2 = result_f1['acc']
    f3 = result_last['acc']

    max_f1 = max(f1, f2, f3)
    max_epoch = 0
    if max_f1 == f1:
        max_result = result_acc
        max_epoch = best_epoch_acc
    elif max_f1 == f2:
        max_result = result_f1
        max_epoch = best_epoch_f1
    elif max_f1 == f3:
        max_result = result_last
        max_epoch = config.max_epochs
    results.append(max_result)
    bepoch.append(max_epoch)




print(f'best_all_Acc: {best_all_acc}   best_all_f1: {best_all_f1}\n')
print(f"时频卷积模块w的最小值： {min_w}")
print(f"时频卷积模块w的最大值： {max_w}")
print(bepoch)
with open(log_path, 'a') as file_handle:
    file_handle.write(f'best_all_Acc: {best_all_acc}   best_all_f1: {best_all_f1}\n')
#存储最终结果
model = Audionet_InfoAttentionClassifier(config).cuda()
current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
data_name = os.path.basename(config.dataset_dir)
if config.pretrain:
    name = data_name + '_pre_' + str(current_date)
else:
    name = data_name + '_nopre_' + str(current_date)
save(name, results, config, model)


