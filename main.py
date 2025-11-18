
import os
import time
import datetime

import torch
import torch.utils
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import argparse
from dataset.dataload import MyDataset, MyDataset_two_audio
from torch.nn import init
import tensorboardX

from models.models_e_c_s import AutoEncoder3
from models.models_emo_con import AutoEncoder2x


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
    parser.add_argument("--lr",
                        type=float,
                        default=0.0005)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.9)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=80)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/home/idal-01/caoyuhao/data/data_preprosess_new/depressed_302.pkl")
    parser.add_argument("--model_dir",
                        type=str,
                        default="/mnt/disk/caoyuhao/save_model/")
    parser.add_argument("--image_dir",
                        type=str,
                        default="/mnt/disk/caoyuhao/image/")
    parser.add_argument("--log_dir",
                        type=str,
                        default="/mnt/disk/caoyuhao/log/")
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str,default='')
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--atpretrained_dir', type=str,default='')
    parser.add_argument('--serpretrained_dir', type=str,default='')
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--rnn', type=bool, default=True)

    return parser.parse_args()

config = parse_args()
os.makedirs(config.model_dir,exist_ok = True)
os.makedirs(config.image_dir,exist_ok = True)
os.makedirs(config.log_dir,exist_ok = True)
###########load model#######################
torch.backends.cudnn.benchmark = True
model = AutoEncoder3(config)
if config.cuda:
    device_ids = [int(i) for i in config.device_ids.split(',')]
    model = model.cuda(device_ids[0])
initialize_weights(model)

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model





###########load data#######################
print('start load data')
train_set = MyDataset_two_audio(config.dataset_dir, mode='train', train_ratio=0.8)
test_set = MyDataset_two_audio(config.dataset_dir, mode='test', train_ratio=0.8)

train_loader = DataLoader(train_set,batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
test_loader = DataLoader(test_set,batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
###########train#######################
print('start train')
writer = tensorboardX.SummaryWriter(comment='model')
total_steps = 0
train_iter = 0
val_iter = 0

start_epoch = config.start_epoch
if config.resume :
    # ATnet resume
    resume = torch.load(config.resume_dir)
    tgt_state = model.state_dict()
    train_iter = resume['train_step']
    val_iter = resume['test_step']
    start_epoch = resume['epoch']
    resume_state = resume['model']
    model.load_state_dict(resume_state)
    print('load resume model')

a = time.time()
best_acc = 0
best_epoch = 0
for epoch in range(start_epoch, config.max_epochs):
    epoch_start_time = time.time()

    acc_1 = 0.0
    acc_2 = 0.0

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        outputs, losses, acces = model.train_func(data)

        losses_values = {k:v.item() for k, v in losses.items()}
        acces_values = {k:v.item() for k, v in acces.items()}

        # record loss to tensorboard
        for k, v in losses_values.items():
            writer.add_scalar(k, v, train_iter)

        acc_1 += acces_values['acc_1']
        acc_2 += acces_values['acc_2']

        loss = sum(losses.values())
        writer.add_scalar('train_loss', loss, train_iter)
        # 获取 test_loader 的长度
        total_batches = len(train_loader)

        if (train_iter % (total_batches // 5) == 0):
            print('epoch:{} [{}/{}] Emo_sim Loss: {:.5f}'.format(epoch + 1, i + 1, len(train_loader),losses['emo_sim'].item()))
            print('Space Loss: {:.5f}'.format(losses['space'].item()))
            print('Recon Loss: {:.5f}'.format(losses['recon1'].item() + losses['recon2'].item()))
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

        # if (train_iter % 2000 == 0): #2000
        #     save_path = os.path.join(config.image_dir+'train/'+str(epoch+1),str(train_iter))
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     model.save_fig(data, outputs, save_path)

        train_iter += 1
    print(f'train Acc 1: {float(acc_1)/(i+1)}, train Acc 2: {float(acc_2)/(i+1)}')
    writer.add_scalar('acc_1',float(acc_1)/(i+1),epoch+1)
    writer.add_scalar('acc_2',float(acc_2)/(i+1),epoch+1)


    print("start to validate, epoch %d" %(epoch+1))

    acc_1_v = 0.0
    acc_2_v = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader):



            outputs, losses, acces = model.val_func(data)

            losses_values = {k:v.item() for k, v in losses.items()}
            acces_values = {k:v.item() for k, v in acces.items()}

        # record loss to tensorboard
            for k, v in losses_values.items():
                writer.add_scalar(k+'_v', v, val_iter)

            acc_1_v += acces_values['acc_1']
            acc_2_v += acces_values['acc_2']

            loss = sum(losses.values())
            writer.add_scalar('test_loss', loss, val_iter)
            # 获取 test_loader 的长度
            total_batches = len(test_loader)
            # 设置打印步数
            print_interval = total_batches //2
            if (val_iter % print_interval == 0):
                print('epoch:{} [{}/{}] Emo_sim Loss: {:.5f}'.format(epoch + 1, i + 1, len(test_loader),losses['emo_sim'].item()))
                print('Space Loss: {:.5f}'.format(losses['space'].item()))
                print('Recon Loss: {:.5f}'.format(losses['recon1'].item() + losses['recon2'].item()))
                print('Self_Recon Loss: {:.5f}'.format(losses['self_recon1'].item() + losses['self_recon2'].item()))
                print('Total Test Loss: {:.5f}'.format(loss.item()))
                print('Time spent: {:.2f} s'.format(time.time() - a))
                print('\n')

            if (val_iter % print_interval == 0): #2000
            #     save_path = os.path.join(config.image_dir+'val/'+str(epoch+1),str(val_iter))
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     model.save_fig(data,outputs,save_path)
                with open(config.log_dir + 'val.txt','a') as file_handle:
                    file_handle.write('epoch:{} [{}/{}] Emo_sim Loss: {:.5f}'.format(epoch + 1, i + 1, len(test_loader),losses['emo_sim'].item()))
                    file_handle.write('Space Loss: {:.5f}'.format(losses['space'].item()))
                    file_handle.write('Recon Loss: {:.5f}'.format(losses['recon1'].item() + losses['recon2'].item()))
                    file_handle.write('Self_Recon Loss: {:.5f}'.format(losses['self_recon1'].item() + losses['self_recon2'].item()))
                    file_handle.write('Total Test Loss: {:.5f}'.format(loss.item()))
                    file_handle.write('Time spent: {:.2f} s'.format(time.time() - a))
                    file_handle.write('\n')

            val_iter += 1
        print(f'test Acc 1: {float(acc_1_v)/ (i+1)}, test Acc 2: {float(acc_2_v)/ (i+1)}\n')
        writer.add_scalar('acc_1_v',float(acc_1_v)/ (i+1), epoch+1)
        writer.add_scalar('acc_2_v',float(acc_2_v)/ (i+1), epoch+1)

        acc = (float(acc_1_v)/ (i+1) + float(acc_2_v)/ (i+1))/2
        acc = round(acc, 2)
        if acc > best_acc:
            if best_epoch != 0:
                os.remove(os.path.join(config.model_dir, str(best_epoch) + "_" + f"pretrain_{best_acc}.pth"))
            best_acc = acc
            best_epoch = epoch+1


        if epoch+1 == best_epoch or epoch+1 ==config.max_epochs:
            torch.save({
                'train_step': train_iter,
                'test_step': val_iter,
                'epoch': epoch,
                'model': model.emo_encoder.state_dict(),
            }, os.path.join(config.model_dir, str(best_epoch) + "_" + f"pretrain_{best_acc}.pth"))

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
torch.save({
                'train_step': train_iter,
                'test_step': val_iter,
                'epoch': epoch,
                'model': model.emo_encoder.state_dict(),
            }, os.path.join(config.model_dir,f"last_pretrain_pressure0123_80_{current_time}.pth"))