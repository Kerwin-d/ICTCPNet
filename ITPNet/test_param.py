import argparse
import torch
import os
import shutil


def test_param(server, epoch, save_img):
    parser = argparse.ArgumentParser(description='')
    opt = parser.parse_args()
    opt.gpu_ids = [0]
    opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    opt.save_img = save_img
    opt.epoch = epoch
    if server:
        opt.SDR_path = r'test_sdr'
        opt.GT_path = r'test_hdr'
        opt.OUT_path = 'LCATNET/results/model_{}'.format(epoch)          #save image

    else:
        opt.SDR_path = r'test_sdr'
        opt.GT_path = r'test_hdr'
        opt.OUT_path = 'LCATNET/results/model_{}'.format(epoch)          #save image
    if not os.path.exists(opt.OUT_path) and opt.save_img:
        os.makedirs(opt.OUT_path)

    opt.load_dir = r'CSRNet/model/model_{}.pth'.format(epoch)
    # opt.load_dir = r'LCATNet/model/model_{}.pth'.format(epoch)
    # opt.load_dir = r'FinetuneNet/model/model_{}.pth'.format(epoch)
    opt.save_dir = opt.load_dir.rsplit("/", 1)[0] + '/psnr.txt'
    print('Test start')
    from Net_test import test
    # from test_csrnet import test
    test(opt)


if __name__ == '__main__':
    for i in range(1, 301, 10):
        test_param(server=False, epoch=i, save_img=True)

