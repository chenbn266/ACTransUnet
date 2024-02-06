
from moudels.backbones.you import Unet
from moudels.backbones.unet_swin import *


class cascade_all_11(nn.Module):
    def __init__(self, c=5, n1=16,n2=32, dropout=0.5, norm='gn',deep_supervision=False, num_classes=3,pretrain_path=None):
        super(cascade_all_11, self).__init__()
        self.deep_supervision=deep_supervision
        self.stage1 = Unet(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage2 = Swin_Unet_od_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,)

    def forward(self, x):
        # print(x.shape)
        x1 = self.stage1(x)
        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)

        if self.training and self.deep_supervision:
            out_all = [x2]
            out_all.append(x1)

            return torch.stack(out_all, dim=1)

        return x2


if __name__ == "__main__":
    import torch as t
    print('-----' * 5)
    a = t.randn(2,5,128,128,128)
    net = cascade_all_11(c=5, n1=16,n2=16, dropout=0.5, norm='gn', num_classes=3,pretrain_path=None)
    out = net(a)
    print(out.shape)

