import argparse
import torch
import os
import shutil


def check_dir(opt):
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)


def train_param(server, step):
    parser = argparse.ArgumentParser(description='')
    opt = parser.parse_args()
    if server:
        opt.train_sdr = r'train_sdr'
        opt.train_hdr = r'train_hdr'
        opt.type = 'PNG'
        opt.num = 0
        opt.gpu_ids = [0]
        opt.num_workers = 8
        opt.batch_size = 8
    else:
        opt.train_sdr = r'train_set_sdr'
        opt.train_hdr = r'train_set_hdr'
        opt.type = 'PNG'
        opt.num = 0
        opt.gpu_ids = [0]
        opt.num_workers = 4
        opt.batch_size = 4

    """CSRNet"""
    if step == 1:
        opt.lr = 2.0e-4
        opt.gamma = 0.5
        opt.epoch_start = 1
        opt.epoch_end = 600
        opt.milestones = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
        opt.save_epoch = 1
        #opt.load_dir = 'CSRNet/model/xx.pth'
        opt.save_dir = 'CSRNet/model'
        opt.loss_file = opt.save_dir + '/loss.txt'
        print('train stage:', step)
        check_dir(opt)
        from Net_CSR import train
        train(opt)

    """LCATNet"""
    if step == 2:
        opt.lr = 2.0e-4
        opt.gamma = 0.5
        opt.epoch_start = 1
        opt.epoch_end = 300
        opt.milestones = [60, 120, 180, 240, 300]
        opt.save_epoch = 1
        opt.load_dir = 'CSRNet/model/xx.pth'
        opt.save_dir = 'LCATNet/model'
        opt.loss_file = opt.save_dir + '/loss.txt'
        print('train stage:', step)
        check_dir(opt)
        from Net_fusion import train
        train(opt)

    """Finetune"""
    if step == 3:
        opt.lr = 2.0e-4
        opt.gamma = 0.5
        opt.epoch_start = 1
        opt.epoch_end = 300
        opt.milestones = [60, 120, 180, 240, 300]
        opt.save_epoch = 1
        opt.load_dir = 'LCATNet/model/xx.pth'
        opt.save_dir = 'Finetune/model'
        opt.loss_file = opt.save_dir + '/loss.txt'
        print('train stage:', step)
        check_dir(opt)
        from Net_finetune import train
        train(opt)


if __name__ == '__main__':
    train_param(server=False, step=1)
