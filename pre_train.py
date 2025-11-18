
import os
import pickle
import random
import time
import datetime

import numpy as np
import torch
import torch.utils
from nvidia import cudnn

import argparse
import tensorboardX
from torch import nn

from dataset.dataload import pre_dataloader
from models.models_e_c import AudioNet_2Encoder
from models.models_e_c_s import AudioNet_3Encoder
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
import os
def tsne_visualize_features_batch(feature1, feature2, feature3, save_path):
    """
    对三个特征的每个batch进行t-SNE降维可视化并保存图片

    参数:
    feature1 (numpy.ndarray): 第一个特征矩阵，形状为(batch_size, seq_len, feature_dim)
    feature2 (numpy.ndarray): 第二个特征矩阵，形状为(batch_size, seq_len, feature_dim)
    feature3 (numpy.ndarray): 第三个特征矩阵，形状为(batch_size, seq_len, feature_dim)
    save_path (str): 保存图片的路径

    返回:
    None
    """
    batch_size = feature1.shape[0]

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(batch_size):
        # 提取当前batch的特征
        f1 = feature1[i].reshape(-1, feature1.shape[-1])
        f2 = feature2[i].reshape(-1, feature2.shape[-1])
        f3 = feature3[i].reshape(-1, feature3.shape[-1])

        # 将三个特征合并为一个大的特征矩阵
        features = np.concatenate((f1, f2, f3), axis=0)

        # 创建一个t-SNE模型，将特征降维到2维
        tsne = TSNE(n_components=2, random_state=0)
        features_tsne = tsne.fit_transform(features)

        # 分离出降维后的三个特征
        feature1_tsne = features_tsne[:f1.shape[0]]
        feature2_tsne = features_tsne[f1.shape[0]:f1.shape[0] + f2.shape[0]]
        feature3_tsne = features_tsne[f1.shape[0] + f2.shape[0]:]

        # 可视化
        plt.figure(figsize=(8, 6))
        plt.scatter(feature1_tsne[:, 0], feature1_tsne[:, 1], label='E_e', alpha=0.6)
        plt.scatter(feature2_tsne[:, 0], feature2_tsne[:, 1], label='E_s', alpha=0.6)
        plt.scatter(feature3_tsne[:, 0], feature3_tsne[:, 1], label='E_c', alpha=0.6)
        plt.title(f't-SNE Visualization of Three Features (Batch {i + 1})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(fontsize='large', title_fontsize='x-large', prop={'weight': 'bold'})
        # plt.show()

        # 保存图片
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f't-SNE图_Batch{i + 1}_{current_time}.pdf'
        plt.savefig(os.path.join(save_path, filename))
        plt.close()


# 示例用法
# 假设你已经有了三个特征矩阵feature1、feature2、feature3，形状为(32, 322, 768)
# save_path = '/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/image/'
# tsne_visualize_features_batch(feature1, feature2, feature3, save_path)



def parse_args():
    parser = argparse.ArgumentParser()
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
                        default=32)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=50)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/home/idal-01/code/TD-learnableFilters/CMDC.pkl")
    parser.add_argument("--model_dir",
                        type=str,
                        default="/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/save_model/")
    parser.add_argument("--image_dir",
                        type=str,
                        default="/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/image/tsne/")
    parser.add_argument("--log_dir",
                        type=str,
                        default="/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/log/")
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument("--flag", type=int, default=1)
    parser.add_argument("--weight", type=float, default=1)
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str,default='')
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.001)
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
    cudnn.benchmark = True


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

f = open(config.dataset_dir, 'rb')

data, Labels = pickle.load(f)
# data = [tensor.squeeze().numpy() for tensor in data]
print('Load data finished' + '\n')
print('The number of patient: ', len(np.where(np.array(Labels) == 1)[0]))
print('The number of NC: ', len(np.where(np.array(Labels) == 0)[0]))

train_loader = pre_dataloader(config, data)

###########train#######################
print('start train')
writer = tensorboardX.SummaryWriter(comment='model')
total_steps = 0
train_iter = 0
a = time.time()
print("build the models ...")


model = AudioNet_3Encoder(config).cuda()
for param in model.spe_encoder.parameters():
    param.requires_grad = False
for param in model.con_encoder.parameters():
    param.requires_grad = False



total_info = sum([param.nelement() for param in model.parameters()])
print('Number of parameter: %.6f' % total_info)

# with open('model_structure.txt', 'w') as file:
#     # 遍历模型的每一层
#     for child in model.children():
#         if isinstance(child, nn.Module):
#             child_str = str(child)
#             file.write('\n' + '=' * 100 + '\n')  # 添加间隔符号
#             file.write(child_str + '\n')

optimizer1 = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=config.lr,
                                  weight_decay=0.001)

optim_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=60, gamma=0.95)



for epoch in range(config.start_epoch, config.max_epochs):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0


    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        outputs, losses = model.process(data)

        losses_values = {k:v.item() for k, v in losses.items()}
        # record loss to tensorboard
        for k, v in losses_values.items():
            writer.add_scalar(k, v, train_iter)

        loss = sum(losses.values())
        model.zero_grad()
        loss.backward()
        optimizer1.step()
        optim_scheduler1.step()

        total_loss += loss


        writer.add_scalar('train_loss', loss, train_iter)
        # 获取 test_loader 的长度
        total_batches = len(train_loader)

        if (train_iter % (total_batches // 4) == 0):
            print('epoch:{} [{}/{}] Emo_sim Loss: {:.5f}'.format(epoch + 1, i + 1, len(train_loader),losses['emo_sim'].item()))
            print('Space Loss: {:.5f}'.format(losses['space'].item()))
            print('Recon Loss: {:.5f}'.format(losses['recon1'].item() + losses['recon2'].item()))
            print('Recon emotion Loss: {:.5f}'.format(losses['recon1_e2'].item() + losses['recon2_e1'].item()))
            print('Self_Recon Loss: {:.5f}'.format(losses['self_recon1'].item() + losses['self_recon2'].item()))
            print('Diff Loss: {:.5f}'.format(losses['diff_1'].item() + losses['diff_2'].item()))
            print('Total Train Loss: {:.5f}'.format(loss.item()))
            print('Time spent: {:.2f} s'.format(time.time() - a))
            print('\n')

        if (train_iter % (total_batches // 4) ==0):

            with open(config.log_dir + 'train.txt','a') as file_handle:
                file_handle.write('epoch:{} [{}/{}] Emo_sim Loss: {:.5f}'.format(epoch+1,i + 1, len(train_loader), losses['emo_sim'].item()))
                file_handle.write('Space Loss: {:.5f}'.format(losses['space'].item()))
                file_handle.write('Recon Loss: {:.5f}'.format(losses['recon1'].item() + losses['recon2'].item()))
                file_handle.write('Self_Recon Loss: {:.5f}'.format(losses['self_recon1'].item() + losses['self_recon2'].item()))
                file_handle.write('Diff Loss: {:.5f}'.format(losses['diff_1'].item() + losses['diff_2'].item()))
                file_handle.write('Total Train Loss: {:.5f}'.format(loss.item()))
                file_handle.write('Time spent: {:.2f} s'.format(time.time() - a))
                file_handle.write('\n')

        train_iter += 1

    model = model.eval()
    with torch.no_grad():
        e1, s1, c1, e2, s2, c2 = model(data)

        e1 = e1.cpu().numpy()
        s1 = s1.cpu().numpy()
        c1 = c1.cpu().numpy()

        e2 = e2.cpu().numpy()
        s2 = s2.cpu().numpy()
        c2 = c2.cpu().numpy()

        save_path = config.image_dir
        tsne_visualize_features_batch(e1, s1, c1, save_path)
        tsne_visualize_features_batch(e2, s2, c2, save_path)
    print('epoch:{} total_train_loss: {:.10f}'.format(epoch + 1, total_loss / len(train_loader)))
    print('\n')
    # current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if (epoch + 1) % 10 == 0:
        torch.save({"emo_encoder": model.emo_encoder.state_dict()},
                   os.path.join(config.model_dir, f"epoch{epoch + 1}_CMDC_tsne.pth"))


