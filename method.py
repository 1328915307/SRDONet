import copy
import time
from collections import OrderedDict, Counter
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm


def evaluate(dataset, MODEL, toPersonList):
    model = MODEL.eval()
    with torch.no_grad():
        pre_labels, tru_labels = [], []

        for i, data in enumerate(dataset):
            inputs = data['input1']
            labels = data['label1']
            pred = model(inputs.cuda())
            pre_label = torch.argmax(pred, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            pre_labels.extend(pre_label)
            tru_labels.extend(labels)

        true_labels, true_targets, true_probs, true_sex = [], [], [], []
        for idx, per in enumerate(toPersonList):
            if idx == 0:
                start = 0
            end = start + per
            #每个人的预测标签
            target_split = pre_labels[start:end]
            true_target = LabelVote(target_split)
            true_targets.append(true_target)
            #每个人的真实标签
            labels_split = tru_labels[start:end]
            true_label = LabelVote(labels_split)
            true_labels.append(true_label)
            start = end

        print('True', tru_labels)
        print('Target', pre_labels)
        print('-----' * 10)
        print('True_labels.', true_labels)
        print('True_targets', true_targets)
        Result = {
            'prec': round(metrics.precision_score(true_labels, true_targets), 5),
            'recall': round(metrics.recall_score(true_labels, true_targets), 5),
            'acc': round(metrics.accuracy_score(true_labels, true_targets), 5),
            'F1': round(metrics.f1_score(true_labels, true_targets), 5),
            'F1_w': round(metrics.f1_score(true_labels, true_targets, average='weighted'), 5),
            'F1_micro': round(metrics.f1_score(true_labels, true_targets, average='micro'), 5),
            'F1_macro': round(metrics.f1_score(true_labels, true_targets, average='macro'), 5),
            'matrix': confusion_matrix(true_labels, true_targets)
        }
        print(Result)
    return Result

def LabelVote(list_):
    counter = Counter(list_)
    majority = counter.most_common(1)[0][0]
    return majority

def save(name, results, args, model):
    path = '/home/idal-01/caoyuhao/Audio_self-supervised_disentanglement_205/results/'
    prec, recall, acc, f1, auc = [], [], [], [], []
    # weight = model.emo_encoder.conv_t_f.t_f_weight

    for i in results:
        prec.append(i['prec'])
        recall.append(i['recall'])
        f1.append(i['F1_w'])
        acc.append(i['acc'])


    prec_mean, prec_sd = np.mean(prec), np.std(prec)
    recall_mean, recall_sd = np.mean(recall), np.std(recall)
    acc_mean, acc_sd = np.mean(acc), np.std(acc)
    f1_mean, f1_sd = np.mean(f1), np.std(f1)


    with open(path + str(name) + '.txt', 'w') as file:
        file.write('Args:\n')
        for arg_name, arg_value in args.__dict__.items():
            file.write(f"{arg_name}: {arg_value}\n")  # 将每个参数名和值写入文件
        file.write('\n' + '=' * 100 + '\n')  # 添加间隔符号
        file.write('\nResults:\n')
        result_str = ""
        for i, result in enumerate(results):
            result_str += f"Result {i + 1}: "
            result_str += f"Precision: {result['prec']:.4f} | "
            result_str += f"Recall: {result['recall']:.4f} | "
            result_str += f"F1: {result['F1_w']:.4f} | "
            result_str += f"Accuracy: {result['acc']:.4f}\n"


        file.write(result_str)
        file.write('\n' + '=' * 100 + '\n')  # 添加间隔符号
        file.write('\nMean+-SD:\n')
        file.write('Prec_mean + SD == %.3f +- %.3f\n' % (prec_mean, prec_sd))
        file.write('Recall_mean + SD == %.3f +- %.3f\n' % (recall_mean, recall_sd))
        file.write('ACC_mean + SD == %.3f +- %.3f\n' % (acc_mean, acc_sd))
        file.write('F1_mean + SD == %.3f +- %.3f\n' % (f1_mean, f1_sd))
        print('ACC_mean + SD == %.3f +- %.3f\n' % (acc_mean, acc_sd))
        print('F1_mean + SD == %.3f +- %.3f\n' % (f1_mean, f1_sd))
        print('Prec_mean + SD == %.3f +- %.3f\n' % (prec_mean, prec_sd))
        print('Recall_mean + SD == %.3f +- %.3f\n' % (recall_mean, recall_sd))

        file.write('\n' + '=' * 100 + '\n')  # 添加间隔符号
        file.write('\nOur Model:\n')
        for child in model.children():
            if isinstance(child, nn.Module):
                child = str(child)
                file.write('\n' + '=' * 100 + '\n')  # 添加间隔符号
                file.write(child + '\n')





























