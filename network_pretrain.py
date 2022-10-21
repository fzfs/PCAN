 # -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models


class PixelWeight(nn.Module):
    def __init__(self, c, weight, num_classes):
        super(PixelWeight, self).__init__()
        self.weight = weight
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pix_predict = nn.Conv2d(c, num_classes, kernel_size=1, stride=1, bias=True)
        
        if self.weight == 1:
            self.ln = nn.LayerNorm(c)
            self.multi_head_attn = nn.MultiheadAttention(c, 8)
            self.pix_weight = nn.Conv2d(c, num_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        # Pixel-wise classification branch
        x1_predict = torch.sigmoid(self.pix_predict(x))
        
        # PCAN
        if self.weight == 1:            
            x2 = self.attention(x)
            
            #  Pixel-wise attention branch
            x2_weight = self.pix_weight(x2)
            x2_weight_normalize = self.norm(x2_weight, x1_predict.size())
            x2_weight_predict = torch.sum(torch.sum(x1_predict * x2_weight_normalize, dim=-1), dim=-1)
            return x1_predict, x2_weight_normalize, x2_weight_predict
        
        # Multiple instance learning
        elif self.weight == 2:
            x2_mil_predict = 1 - torch.exp(torch.sum(torch.sum(torch.log(self.norm_s(1 - x1_predict)), dim=-1), dim=-1))
            
            return x1_predict, x2_mil_predict
        
        # Global average pooling
        else:
            x2_avg_predict = self.pool(x1_predict).view(x.size(0), -1)
            return x1_predict, x2_avg_predict

    def attention(self, x):
        shape_input = x.size()
        x1 = x.view(shape_input[0], shape_input[1], shape_input[2] ** 2)
        x1 = x1.permute(2, 0, 1)

        x1_temp, _ = self.multi_head_attn(x1, x1, x1)
        x1 = x1 + x1_temp
        x1 = self.ln(x1)

        x1 = x1.permute(1, 2, 0)
        x1 = x1.view(shape_input)
        return x1

    def norm(self, x, shape):
        x = x.view(shape[0], shape[1], shape[2] ** 2)
        x = torch.softmax(x, dim=-1)
        x = x.view(shape)
        return x
    
    def norm_s(self, x):
        x = x * 0.02 + 0.98
        return x


class Densenet121(nn.Module):

    def __init__(self, pretrain=True, avg=0, weight=1, truncated=0, num_classes=5):

        super(Densenet121, self).__init__()
        self.avg = avg
        self.model = models.densenet121(pretrained=pretrain).features
        C = 1024

        if truncated == 1:
            C = 512
            del self.model.norm5
            del self.model.denseblock4
            del self.model.transition3.pool
            self.model.add_module('norm5', nn.BatchNorm2d(C))

        self.relu = nn.ReLU(inplace=True)

        if avg == 0:
            self.PixelWeight_t3 = PixelWeight(C, weight, num_classes)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.relu(x)

        if self.avg == 0:  
            return self.PixelWeight_t3(x)

        else:
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.sigmoid(self.fc(x))

            return x
