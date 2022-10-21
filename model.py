# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from networks import define_network
from torch.optim import lr_scheduler
import functools

class Net(nn.Module):
    def __init__(self, lr, weight_decay, init_type, gpu_ids, network,
                 pretrain, avg, weight, milestones, truncated, alpha, num_classes):
        super(Net, self).__init__()
        self.lr = lr
        self.avg = avg
        self.alpha = alpha
        self.weight = weight
        self.gpu_ids = gpu_ids
        self.network = network
        self.truncated = truncated
        self.milestones = milestones

        self.model = define_network(init_type, gpu_ids, network, pretrain, avg, weight, truncated, num_classes)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=weight_decay, momentum=0.9)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=0.1, last_epoch=-1)

        self.loss1 = 0
        self.loss2 = 0
        self.loss = 0

    def forward(self, mode='train'):
        if self.avg == 0:
            # PCAN
            if self.weight == 1:
                self.predicted_all, self.weight_all, self.final_predicted = self.model(self.im)
                self.loss = self.criterion(self.final_predicted.float(), self.label.float())
            
            # Multiple instance learning
            else:
                self.predicted_all, self.final_predicted = self.model(self.im)                
                self.weight_all = self.predicted_all
                self.loss = self.criterion(self.final_predicted.float(), self.label.float())

        # Global average pooling
        else:
            self.final_predicted = self.model(self.im)
            self.loss = self.criterion(self.final_predicted.float(), self.label.float())

    def set_input(self, x):
        self.im = x['im'].cuda()
        self.label = x['label'].cuda()

        self.w = self.im.size()[-1] // 32 if self.truncated == 1 else self.im.size()[-1] // 16

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def predicted_val(self):
        a = self.final_predicted.cpu().numpy()
        b = self.label.cpu().numpy()
        return a, b

    def predicted_test(self):
        a = self.final_predicted.cpu().numpy()
        b = self.predicted_all.cpu().numpy()
        c = self.label.cpu().numpy()
        d = self.weight_all.cpu().numpy()
        e = self.im.cpu().numpy()

        return a, b, c, d, e

    def update_learning_rate(self):
        self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

    def print_networks(self):
        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        print('Total number of parameters : %.3f M' % (num_params / 1e6))
