import CSRNet.CSRNet_arch as CSRNet_arch
import torch.nn as nn
import torch

from datasets.ICTCP_convert import SDR_to_ICTCP, ICTCP_to_HDR


class WholeNet(nn.Module):
    def __init__(self, device,num=3, channel=32):
        super(WholeNet, self).__init__()
        self.CSRNet_I = CSRNet_arch.CSRNet(in_nc=1, out_nc=1, base_nf=64, cond_nf=32)
        self.CSRNet_TP = CSRNet_arch.CSRNet(in_nc=2, out_nc=2, base_nf=64, cond_nf=32)

    def forward(self, sdrRGB):
        sdrITP = SDR_to_ICTCP(sdrRGB,dim=1)
        sdrI, sdrT, sdrP = torch.split(sdrITP, 1, dim=1)
        sdrTP = torch.cat([sdrT, sdrP], dim=1)
        I = self.CSRNet_I(sdrI)
        TP = self.CSRNet_TP(sdrTP)

        ITP = torch.cat([I, TP], dim=1)
        hdrRGB = ICTCP_to_HDR(ITP,dim=1)

        return [ITP, hdrRGB]


