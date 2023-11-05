#!/usr/bin/env python
# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader
import torch.utils.data as data
#import h5py
import torch
import numpy as np
import os
import cv2
from datasets.ICTCP_convert import SDR_to_ICTCP, HDR_to_ICTCP


class PNG_dataset(data.Dataset):
    def __init__(self, sdr_dir, gt_dir, num, is_random=True):
        super(PNG_dataset, self).__init__()
        self.sdr_list = []
        self.gt_list = []
        name_list = os.listdir(sdr_dir)
        for name in name_list:
            self.sdr_list.append(sdr_dir + '/' + name)
            self.gt_list.append(gt_dir + '/' + name)
        if is_random:
            rnd_index = np.arange(len(self.sdr_list))
            np.random.shuffle(rnd_index)
            self.sdr_list = np.array(self.sdr_list)[rnd_index]
            self.gt_list = np.array(self.gt_list)[rnd_index]
        if num != 0:
            self.sdr_list = self.sdr_list[:num]
            self.gt_list = self.gt_list[:num]

    def __getitem__(self, index):
        #print(self.sdr_list)
        #print(self.gt_list)
        input_ = cv2.imread(self.sdr_list[index], flags=-1)[:,:,::-1]                #宽x高x通道数
        target_ = cv2.imread(self.gt_list[index], flags=-1)[:,:,::-1]                #cv.imread()读取通道的顺序三BGR，[:,:,::-1]转换为RGB

        input_ = np.array(input_, np.float32) / 255
        target_ = np.array(target_, np.float32) / 65535

        sdrRGB = torch.from_numpy(input_).float().permute(2, 0, 1)                   #通道数x宽x高
        gtRGB = torch.from_numpy(target_).float().permute(2, 0, 1)

        sdrITP = SDR_to_ICTCP(sdrRGB,dim=0)
        gtITP = HDR_to_ICTCP(gtRGB,dim=0)

        return {'sdrRGB': sdrRGB, 'gtRGB': gtRGB,
                'sdrITP': sdrITP, 'gtITP': gtITP}

    def __len__(self):
        return len(self.sdr_list)


class H5_dataset(data.Dataset):
    def __init__(self, file, num):
        super(H5_dataset, self).__init__()
        with h5py.File(file, 'r') as f:
            if num != 0:
                self.sdr = f['sdr'][:num]
                self.hdr = f['hdr'][:num]
            else:
                self.sdr = f['sdr'][()]
                self.hdr = f['hdr'][()]

    def __getitem__(self, index):
        sdr = self.sdr[index].astype('float32') / 255.0
        hdr = self.hdr[index].astype('float32') / 65535.0

        sdrRGB = torch.from_numpy(sdr).float().permute(2, 0, 1)
        gtRGB = torch.from_numpy(hdr).float().permute(2, 0, 1)

        sdrITP = SDR_to_ICTCP(sdrRGB, dim=0)
        gtITP = HDR_to_ICTCP(gtRGB, dim=0)

        return {'sdrRGB': sdrRGB, 'gtRGB': gtRGB,
                'sdrITP': sdrITP, 'gtITP': gtITP}

    def __len__(self):
        return len(self.sdr)


def create_dataset(opt):
    train_set = PNG_dataset(opt.train_sdr, opt.train_hdr, opt.num)
    print('--PNG数据加载完成')
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)        #shuffle:在每个epoch开始的时候，对数据进行重新排序
    return train_loader


