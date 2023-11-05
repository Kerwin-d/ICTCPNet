from time import time
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from tensorboardX import SummaryWriter
from torch.nn import init

from datasets.dataset_ITP import create_dataset
from model_fusion import WholeNet


class Net:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 第一个GPU
        self.lr = opt.lr
        self.gamma = opt.gamma
        self.milestones = opt.milestones
        self.save_dir = opt.save_dir
        self.load_dir = opt.load_dir
        self.tarin_status()


    def tarin_status(self):
        self.model = WholeNet(self.device).to(self.device)
        self.set_loss_optimizer_scheduler()
        self.load_network()
        if self.gpu_ids:
            self.model = nn.DataParallel(self.model, self.gpu_ids)  # 模型迁移到多个GPU上
        self.model.train()

    def set_loss_optimizer_scheduler(self):
        self.L1 = nn.L1Loss().to(self.device)
        self.optim1 = optim.Adam(self.model.fusionNet.parameters(), lr=self.lr)
        self.sche1 = lr_scheduler.MultiStepLR(self.optim1, milestones=self.milestones, gamma=self.gamma)
        self.optimizers = [self.optim1]
        self.schedulers = [self.sche1]


    def load_network(self):
         self.init_weight(self.model, 'xavier')
         if self.load_dir is not None:
             checkpoint = torch.load(self.load_dir, map_location=self.device)
             self.model.CSRNet_I.load_state_dict(checkpoint['CSRNet_I'])
             self.model.CSRNet_TP.load_state_dict(checkpoint['CSRNet_TP'])
             #self.model.fusionNet.load_state_dict(checkpoint['fusionNet'])
             print('--完成权重加载:{}--'.format(self.load_dir))


    def init_weight(self,net, init_type):
            def init_func(m):
                classname = m.__class__.__name__
                if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    if init_type == 'normal':
                        init.normal_(m.weight.data)
                    elif init_type == 'xavier':
                        init.xavier_normal_(m.weight.data)
                    elif init_type == 'kaiming':
                        init.kaiming_normal_(m.weight.data)
                    elif init_type == 'orthogonal':
                        init.orthogonal_(m.weight.data)
                    else:
                        raise NotImplementedError('initialization method {} is not implemented'.format(init_type))
                elif classname.find('BatchNorm2d') != -1:
                    init.normal_(m.weight.data)
                    init.constant_(m.bias.data, 0.0)

            print('--initialize network with {}'.format(init_type))
            net.apply(init_func)


    def get_current_lr(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups][0]

    def schedulers_step(self):
        for sche in self.schedulers:
            sche.step()

    def save_network(self, epoch):
        save_path = self.save_dir + '/model_{}.pth'.format(epoch)
        state = {
                 'CSRNet_I': self.model.module.CSRNet_I.state_dict(),
                 'CSRNet_TP': self.model.module.CSRNet_TP.state_dict(),
                 'fusionNet': self.model.module.fusionNet.state_dict(),
                 }
        torch.save(state, save_path)

    def train_step(self, data):
        for optim in self.optimizers:
            optim.zero_grad()

        """set data"""
        self.sdrRGB = data['sdrRGB'].to(self.device)
        self.sdrITP = data['sdrITP'].to(self.device)
        self.gtRGB = data['gtRGB'].to(self.device)
        self.gtITP = data['gtITP'].to(self.device)
        """cal loss"""
        self.hdrITP1, self.hdrRGB1 = self.model(self.sdrRGB)

        self.loss = self.L1(self.hdrITP1, self.gtITP)                             #ITP空间loss
        self.psnr_ITP = kornia.psnr_loss(self.hdrITP1, self.gtITP, max_val=1)
        self.psnr_RGB = kornia.psnr_loss(self.hdrRGB1, self.gtRGB, max_val=1)

        """back"""
        self.loss.backward()
        for optim in self.optimizers:
            optim.step()

    def tensorboard(self):
        loss = self.loss.item()
        psnr_ITP = self.psnr_ITP.item()
        psnr_RGB = self.psnr_RGB.item()

        return loss, psnr_ITP,psnr_RGB



def train(opt):
    torch.manual_seed(901)
    train_loader = create_dataset(opt)
    print("数据加载完成")
    batch_num = len(train_loader)
    model = Net(opt)

    for epoch in range(opt.epoch_start, opt.epoch_end + 1):
        print("开始训练")
        losses = []
        psnres_ITP = []
        psnres_RGB = []

        start = time()
        lr = model.get_current_lr()
        for i, data in enumerate(train_loader, 1):
            model.train_step(data)

            loss,psnr_ITP,psnr_RGB = model.tensorboard()

            losses.append(loss)
            psnres_ITP.append(psnr_ITP)
            psnres_RGB.append(psnr_RGB)

            if i % 10 == 0:
                print('epoch:%d, batch:%d/%d, lr:%.7f,   '
                      'loss:%.6f, psnr_ITP1:%.2f, psnr_RGB1:%.2f, \n'
                      % (epoch, i, batch_num, lr, loss, psnr_ITP, psnr_RGB))
        epoch_message = 'epoch:%d, batch_size:%d, lr:%.7f, time:%d, ' \
                        'loss:%.6f, psnr_ITP1:%.2f, psnr_RGB1:%.2f' \
                        % (epoch, opt.batch_size, lr, (time() - start) / 60,
                           np.mean(losses), np.mean(psnres_ITP), np.mean(psnres_RGB))

        with open(opt.loss_file, 'a', encoding='utf-8') as f:
            f.write(epoch_message)
            f.write('\n')
        print(epoch_message)
        print('------------')
        model.schedulers_step()
        if epoch % opt.save_epoch == 0:
            model.save_network(epoch=epoch)

