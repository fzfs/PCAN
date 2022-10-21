# -*- coding: utf-8 -*-

import os
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyDataset(Dataset):
    def __init__(self, csv_path, mode, class_name, image_size):
        self.mode = mode
        self.class_name = class_name
        self.image_size = image_size
        data = pd.read_csv(csv_path)
        labels = data[class_name]
        labels = labels.fillna(value=0)
        labels = labels.replace(-1, 1)
        labels = labels.values
        imgs = []
        for index, row in data.iterrows():
            label = labels[index, :]
            imgs.append((row['Path'], label))
            
        self.imgs = imgs
        
    def __getitem__(self, index):
        path, label = self.imgs[index]
        im = Image.open(path).convert('L')
        im = im.convert('RGB')
        
        transform = get_transform(self.mode, self.image_size)
        im = transform(im)
        return {'im': im, 'label': label}

    def __len__(self):
        return len(self.imgs)

def get_transform(mode='train', image_size=224):
    transform_list = []
    
    if image_size == 224:        
        transform_list = [transforms.Resize(256)]
        if mode == 'train':
            transform_list += [transforms.RandomCrop(224)]
            transform_list += [transforms.RandomHorizontalFlip()]
            
        else:
            transform_list += [transforms.CenterCrop(224)]
    else:
        if mode == 'train':
            transform_list += [transforms.RandomHorizontalFlip()]
    
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        
    return transforms.Compose(transform_list)
