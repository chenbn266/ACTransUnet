from functools import reduce, lru_cache
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from moudels.backbones.swintransformer import SwinTransformerSys3D
import copy
from moudels.backbones.ODconv3D import ODConv3d
from torch.nn.functional import interpolate
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


class ConvD_od(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False, kernel_num=4):
        super(ConvD_od, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = ODConv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, norm=norm,
                              kernel_num=4)
        self.conv2 = ODConv3d(planes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, norm=norm,
                              kernel_num=4)

        self.bn1 = normalization(planes, norm)
        self.bn2 = normalization(planes, norm)


    def forward(self, x):
        if not self.first:
            # print("maxpolo",x.shape)
            x = self.maxpool(x)
            # print("maxpolo",x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        # print("sfae",x1.shape)
        x = self.relu(self.bn2(self.conv2(x)))
        # print("sfae", x2.shape)
        return x


class ConvU_od(nn.Module):
    def __init__(self, planes, norm='gn', first=False, kernel_num=4):
        super(ConvU_od, self).__init__()

        self.first = first
        self.relu = nn.LeakyReLU(inplace=True)

        if not self.first:
            self.conv1 = nn.Conv3d(2 * planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = normalization(planes, norm)
        self.conv2 = ODConv3d(planes, planes // 2, kernel_size=1, stride=1, padding=0, reduction=0.0625, kernel_num=4)
        self.conv3 = ODConv3d(planes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, kernel_num=4)
        self.bn2 = normalization(planes // 2, norm)
        self.bn3 = normalization(planes, norm)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.cat([prev, x], 1)
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


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
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.cat([prev, x], 1)
        x = self.relu(self.bn3(self.conv3(x)))

        return x




class Swin_Unet_od_9(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', norm_layer=nn.LayerNorm, num_classes=3, pretrain=None):
        super(Swin_Unet_od_9, self).__init__()

        self.convd0 = ConvD(c, n, dropout, norm, first=True)
        self.convd1 = ConvD_od(n, 2 * n, dropout, norm)
        self.convd2 = ConvD_od(2 * n, 4 * n, dropout, norm)

        self.convu2 = ConvU_od(4 * n, norm, True)
        self.convu1 = ConvU_od(2 * n, norm, )

        self.seg = nn.Conv3d(2 * n, num_classes, 1)


        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(32, 32, 32),
            patch_size=(4, 4, 4),
            in_chans=n * 4,
            num_classes=n * 4,
            embed_dim=192,
            depths=[2, 2, 1],
            depths_decoder=[1, 2, 2],
            num_heads=[6, 12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=norm_layer,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        x2 = self.convd2(x1)
        y = self.swin(x2)
        y2 = self.convu2(y, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")




if __name__ == "__main__":
    import torch as t

    print('-----' * 5)
    rga = t.randn(2, 5, 128, 128, 128)
    net = Swin_Unet_od_9(c=5, n=32, dropout=0.5, norm='gn', num_classes=3)
    print(net)
    out = net(rga)
    print(out[0].shape,out[1].shape)
