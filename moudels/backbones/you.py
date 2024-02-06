import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from moudels.isnet.imagelevel import *
from moudels.backbones.swintransformer import SwinTransformerSys3D, SwinTransformerSys3D_encoder


def trilinear_kernel(in_channels, out_channels, kernel_size):
    """Define a bilinear kernel according to in channels and out channels.
    Returns:
        return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor) * (
            1 - abs(og[2] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :, :] = bilinear_filter
    return torch.from_numpy(weight)


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)

    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)

        self.bn1 = normalization(planes, norm)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))

        return x2


class ConvU(nn.Module):
    def __init__(self, planes, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.first = first
        self.relu = nn.LeakyReLU(inplace=True)

        if not self.first:
            self.conv1 = nn.Conv3d(2 * planes, planes, 3, 1, 1, bias=False)
            self.bn1 = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes // 2, 1, 1, 0, bias=False)
        self.bn2 = normalization(planes // 2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3 = normalization(planes, norm)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
            # print("upx",x.shape)
        # print("upqian",x.shape,prev.shape)
        y = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        # print("uphou", y.shape)
        y = self.relu(self.bn2(self.conv2(y)))
        # print("upyy", y.shape)
        y = torch.cat([prev, y], 1)
        # print("upc", y.shape)
        y = self.relu(self.bn3(self.conv3(y)))
        # print("upend", y.shape)

        return y


class Unet(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4, deep_supervision=False):
        super(Unet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision = deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)

        self.seg_pred = nn.Conv3d(16 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x3 = self.convd3(x2)

        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        out = self.seg1(y1)

        return out


if __name__ == "__main__":
    import torch as t

    print('-----' * 5)

    a = t.randn(2, 5, 128, 128, 128)
    deeps = [16, 32, 64, 128, 256]
    net = Unet(c=5, n=32, dropout=0.5, norm='gn', num_classes=3, deep_supervision=True)
    out = net(a)
    print(out.shape)
