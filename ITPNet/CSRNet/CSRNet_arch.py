import functools
import math
import torch
import torch.nn as nn



class Condition(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super(Condition, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)
        return out


# 3layers with control
class CSRNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_nf=64, cond_nf=32):
        super(CSRNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)

        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True)

        self.cond_scale2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)

        self.cond_scale3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = self.cond_net(x)

        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        out = self.conv1(x)
        out = out * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv2(out)
        out = out * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv3(out)
        out = out * scale3.view(-1, self.out_nc, 1, 1) + shift3.view(-1, self.out_nc, 1, 1) + out
        return out


# model = CSRNet(in_nc=1, out_nc=1, base_nf=64, cond_nf=32)
# d1 = torch.rand(1, 1, 480, 480)
#
# o = model(d1)
# print()
# onnx_path = "onnx_model_name.onnx"
# torch.onnx.export(model, d1, onnx_path)
# netron.start(onnx_path)
