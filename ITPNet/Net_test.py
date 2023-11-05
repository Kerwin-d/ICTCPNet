import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from datasets.ICTCP_convert import SDR_to_ICTCP, HDR_to_ICTCP,ICTCP_to_HDR
from model_fusion import WholeNet


class Net_test:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 第一个GPU
        self.load_dir = opt.load_dir
        self.test_status()

    def test_status(self):
        self.model = WholeNet().to(self.device)

        checkpoint = torch.load(self.load_dir, map_location=self.device)
        self.model.CSRNet_I.load_state_dict(checkpoint['CSRNet_I'])
        self.model.CSRNet_TP.load_state_dict(checkpoint['CSRNet_TP'])
        self.model.fusionNet.load_state_dict(checkpoint['fusionNet'])
        print('--完成权重加载:{}--'.format(self.load_dir))
        if self.gpu_ids:
            self.model = nn.DataParallel(self.model, self.gpu_ids)  # 模型迁移到多个GPU上
        self.model.eval()

def compute_psnr(gt, hdr, peak=1.0):
    mse = np.mean(np.square(gt - hdr))
    psnr = 10 * np.log10(peak * peak / mse)
    return psnr


def test(opt):
    torch.manual_seed(901)
    cudnn.benchmark = True
    save_dir = opt.save_dir
    epoch = opt.epoch
    model = Net_test(opt)

    psnr_ITP_sum = []
    psnr_RGB_sum = []
    list = os.listdir(opt.SDR_path)
    #list = list[0:2]
    for name in tqdm(list):
        sdr_file = opt.SDR_path + '/' + name
        gt_file = opt.GT_path + '/' + name

        ERGB_sdr = cv2.imread(sdr_file, flags=-1)[:, :, ::-1] / 255
        ERGB_sdr = torch.from_numpy(ERGB_sdr).float().permute(2, 0, 1).unsqueeze(0).to(opt.device)

        ERGB_gt = cv2.imread(gt_file, flags=-1)[:, :, ::-1] / 65535
        EITP_gt = HDR_to_ICTCP(torch.from_numpy(ERGB_gt)).cpu().numpy()
        #name = name.split(".")[0] + '_i.png'
        with torch.no_grad():
            csr_result, fusion_result = model.model(ERGB_sdr)
            # fusion_result = model.model(ERGB_sdr)
            ERGB_output1 = fusion_result[1].squeeze(0).permute(1, 2, 0).cpu().numpy()
            #ERGB_output2 = hdrRGB1.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if opt.save_img:
                out_file = opt.OUT_path + '/' + name
                cv2.imwrite(out_file,
                            np.round(np.clip(ERGB_output1, a_min=0, a_max=1) * 65535)[:, :, ::-1].astype('uint16'))

            ERGB_hdr = np.round(np.clip(ERGB_output1, a_min=0, a_max=1) * 65535) / 65535
        EITP_hdr = HDR_to_ICTCP(torch.from_numpy(ERGB_hdr)).cpu().numpy()
        psnr_ITP = compute_psnr(EITP_gt, EITP_hdr)
        psnr_RGB = compute_psnr(ERGB_gt, ERGB_hdr)


        psnr_ITP_sum.append(psnr_ITP)
        psnr_RGB_sum.append(psnr_RGB)
    print()
    print(np.mean(psnr_ITP_sum))
    print(np.mean(psnr_RGB_sum))


    epoch_message = 'epoch:%d' \
                    'psnr_ITP:%.6f, psnr_RGB:%.6f'\
                    % (epoch,
                       np.mean(psnr_ITP_sum), np.mean(psnr_RGB_sum))

    with open(save_dir, 'a', encoding='utf-8') as f:
        f.write(epoch_message)
        f.write('\n')

