# -*- coding: utf-8 -*-

import os
import cv2
import time
import torch
import random
import argparse
import numpy as np
import seaborn as sns
from torch.nn import init
from sklearn import metrics
import matplotlib.pyplot as plt
from MyDataset import MyDataset

def cprint(s, log_file):
    print(s)
    with open(log_file, 'a+') as opt_file:
        print(s, file=opt_file)

def init_net(net, pretrain, init_type, gpu_ids, init_gain=0.02):
    assert(torch.cuda.is_available())
    if len(gpu_ids) > 1:                
        net = torch.nn.DataParallel(net, gpu_ids)
    net.to('cuda')
    if not pretrain:
        init_weights(net, init_type, init_gain=init_gain)
    else:
        print('initialize network with pretrained net')
    return net


def init_weights(net, init_type, init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def print_metrics1(labels_all, predict_all, mode, name, class_name='', auc_loss='auc'):
    aucs = []
    message = []
    log_file = name + '/log.txt'
    for i in range(len(class_name)):
        auc = metrics.roc_auc_score(labels_all[:, i], predict_all[:, i])
        aucs.append(auc)

        temp = class_name[i][0:4] + ' : ' + str(np.around(auc, decimals=4))
        if mode != '':
            cprint(temp, log_file)

        message.append(temp)

    mean = 'Mean : ' + str(np.around(np.mean(np.array(aucs)), decimals=4))
    cprint(mean, log_file)
    message.append(mean)

    if mode == 'test':
        file_name = name+'/'+auc_loss+'_result.txt'
        cprint('\n'.join(message)+'\n', file_name)
    
    return np.mean(np.array(aucs))


def print_metrics(labels_all, predict_all, w, fn):
    mm = np.zeros((w, w))
    for m in range(w):
        for n in range(w):
            aucs = []
            for i in range(14):
                auc = metrics.roc_auc_score(labels_all[:, i], predict_all[:, i, m, n])
                aucs.append(auc)
            mm[m, n] = np.mean(np.array(aucs))
    ax = sns.heatmap(np.array(mm))
    plt.savefig(fn+'/location_auc.png')
    plt.close()
    cprint(mm, fn + '/log.txt')


def visualization(predict, label, img, fn, class_name):
    for i in range(len(label)):
        for j in range(8):
            temp_img = predict[i, j, :, :]
            temp_img = cv2.resize(temp_img, (512, 512))
            temp_img = temp_img - np.min(temp_img)
            temp_img = temp_img / np.max(temp_img)
            temp_img = cv2.applyColorMap(np.uint8(255 * temp_img), cv2.COLORMAP_JET)
            temp_img = np.float32(temp_img) / 255

            temp_input = img[i]
            temp_input = np.transpose(temp_input, [1, 2, 0])
            temp_input = temp_input - np.min(temp_input)
            temp_input = temp_input / np.max(temp_input)

            cam = temp_img + temp_input
            cam = cam / np.max(cam)

            temp_file = fn + '/visualization_weight/' + str(i)

            if not os.path.exists(temp_file):
                os.makedirs(temp_file)

            cv2.imwrite(temp_file + '/' + class_name[j] + "_heatmap.jpg", np.uint8(255 * temp_img))
            cv2.imwrite(temp_file + '/' + class_name[j] + ".jpg", np.uint8(255 * cam))


def parser_model(mode='train'):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--avg', type=int, default=-1, help='avg==1, Global average pooling')
    parser.add_argument('--weight', type=int, default=-1, help='weight== 1, PCAN; weight== 2, Multiple instance learning')
    parser.add_argument('--alpha', type=str, default='1')
    parser.add_argument('--truncated', type=int, default=0, help='truncated=1, truncated DenseNet121')
    parser.add_argument('--auc_loss', type=str, default="auc")
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--network', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=224, help='input size')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--lr', type=str, default='1e-2')
    parser.add_argument('--weight_decay', type=str, default='1e-4')
    parser.add_argument('--tm', type=str, default='337843', help='a part of the file name')
    parser.add_argument('--target_layer_names', type=str, default='relu')
    parser.add_argument('--pretrain', type=int, default=1)    
    parser.add_argument('--milestones', type=str, default='100')
    parser.add_argument('--class_name', type=str, default='No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices')
    
    opt = parser.parse_args()
    opt.alpha = float(opt.alpha)
    opt.lr = float(opt.lr)
    opt.pretrain = bool(opt.pretrain)
    opt.weight_decay = float(opt.weight_decay)
    opt.class_name = opt.class_name.split(',')
    opt.milestones = [int(g) for g in opt.milestones.split(',')]

    if mode == 'train':
        fn = 'runs/' + str(opt.network) + '_' + str(opt.lr) + '_' + str(opt.batch_size) + '_' + str(
            opt.avg) + '_' + str(opt.weight) + '_' + str(opt.truncated) + '_' + str(opt.alpha) + '_' + str(time.time())[-6:]

        if not os.path.exists(fn):
            os.makedirs(fn)

        config = 'lr:' + str(opt.lr) + '\n' + 'avg:' + str(opt.avg) + '\n' + 'weight:' + str(
            opt.weight) + '\n' + 'epochs:' + str(opt.epochs) + '\n' + 'network:' + str(
            opt.network) + '\n' + 'pretrain:' + str(opt.pretrain) + '\n' + 'init_type:' + str(
            opt.init_type) + '\n' + 'image_size:' + str(opt.image_size) + '\n' + 'batch_size:' + str(
            opt.batch_size) + '\n' + 'milestones:' + str(opt.milestones) + '\n' + 'weight_decay:' + str(
            opt.weight_decay) + '\n' + 'truncated:' + str(opt.truncated) + '\n' + 'alpha:' + str(opt.alpha)

        with open(fn + '/config.txt', 'a+') as config_file:
            config_file.write(config)

    else:
        fn = 'runs/' + str(opt.network) + '_' + str(opt.lr) + '_' + str(opt.batch_size) + '_' + str(
            opt.avg) + '_' + str(opt.weight) + '_' + str(opt.truncated) + '_' + str(opt.alpha) + '_' + str(opt.tm)

    return opt, fn

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True


def load_data(path, mode, batch_size, image_size, name, drop_last=False):
    data = MyDataset(path, mode, name, image_size)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=drop_last)
    if mode == 'test':
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=drop_last)

    dataset_size = len(loader)
    print('The number of '+mode+' batches = %d' % dataset_size)
    return loader, dataset_size
