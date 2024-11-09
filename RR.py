import torch
import torch.nn as nn
from torch.nn import init as init
from pcd_align import *
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Soft_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, ks = 3, is_bias = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=ks, padding=int(ks / 2), bias= is_bias),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=ks, padding=int(ks / 2), bias= is_bias),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=ks, padding=int(ks / 2), bias= is_bias),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv(x)
class Res_block(nn.Module):
    def __init__(self, feat_num = 32, ks = 3, is_bias = True):
        super().__init__()
        self.conv1 = nn.Conv2d(feat_num, feat_num, ks, 1, 1, bias=is_bias)
        self.conv2 = nn.Conv2d(feat_num, feat_num, ks, 1, 1, bias=is_bias)
        self.relu = nn.ReLU(inplace=True)
        default_init_weights([self.conv1, self.conv2], 0.1)
    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out
def make_layer(block, feat_num = 32, block_num = 16):
    layers = []
    for i in range(block_num):
        layers.append(block(feat_num = feat_num))
    return nn.Sequential(*layers)
class Conv_Res(nn.Module):
    def __init__(self, in_channels, out_channels=64, block_num=16, is_conv = True):
        super().__init__()

        main = []
        if is_conv == True:
            main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
            main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        # residual blocks
        main.append(make_layer(Res_block, out_channels, block_num))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)
class Net(nn.Module):
    def __init__(self, wins = 41, feat_num = 64, block_num = 16, bilinear=True, kinds = 2):
        super(Net, self).__init__()
        self.feat_num = feat_num
        self.block_num = block_num
        self.bilinear = bilinear
        self.kinds = 2
        self.spk2fe = nn.Sequential(
            nn.Conv2d(wins, feat_num, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.tfi2fe = nn.Sequential(
            nn.Conv2d(1, feat_num, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.att_fe = Soft_Attention(2 * feat_num, 1, feat_num)#fe att
        self.fuse = Conv_Res(feat_num, feat_num, block_num, is_conv = False)
        self.align_img = Easy_PCD(feat_num)
        self.extract_fe_for = Conv_Res(2 * feat_num, feat_num, block_num)
        self.extract_fe_back = Conv_Res(2 * feat_num, feat_num, block_num)
        self.out_from_prop = nn.Sequential(
            DoubleConv(feat_num * 2, feat_num, feat_num * 2),
            nn.Conv2d(feat_num, 1, 3, 1, 1, bias=True)
        )
    def forward(self, spk, tfi = 0, is_prop = True):
        """
        spk (Tensor): Input LR sequence with shape (n, t, wins, h, w).
        tfi (Tensor): Input LR sequence with shape (n, t, h, w).
        """
        n, t, wins, h, w = spk.size()
        print("SPK SIZE is", spk.size())
        fe_spk = self.spk2fe(spk.reshape(-1, wins, h, w))
        print(tfi.size())
        fe_tfi = self.tfi2fe(tfi.reshape(-1, 1, h, w))
        att_fe = self.att_fe(torch.cat([fe_spk, fe_tfi], 1))
        rep = fe_spk * att_fe + fe_tfi * (1 - att_fe)
        fe_init = self.fuse(rep)
        fe_init = fe_init.reshape(n, t, -1, h, w)
        output = []
        fe_imgs = []
        fe_prop = spk.new_zeros(n, self.feat_num, h, w)
        for i in range(t):
            if i == 0:
                #print(img_init.shape)
                fe_t = fe_init[:,i,:,:,:]
            else:
                fe_t, L1_offset, L2_offset, L3_offset, offset = self.align_img(fe_prop, fe_init[:,i,:,:,:])
                fe_t = fe_t + fe_init[:,i,:,:,:]
            if is_prop == True:
                fe_prop = torch.cat([fe_t, fe_prop], dim=1)
                fe_prop = self.extract_fe_for(fe_prop)
                output.append(fe_prop)
        #backward
        fe_prop = spk.new_zeros(n, self.feat_num, h, w)
        imgs_from_prop = []
        imgs_from_align = []
        imgs = []
        for i in range(t-1, -1, -1):
            #print("back is", i)
            if i == t - 1:
                fe_t = fe_init[:,i,:,:,:]
            else:
                fe_t, L1_offset, L2_offset, L3_offset, offset = self.align_img(fe_prop, fe_init[:,i,:,:,:])
                fe_t = fe_t + fe_init[:,i,:,:,:]
            if is_prop == True:
                #if i != t - 1:
                #    fe_prop = self.align_fe(fe_prop, L1_offset, L2_offset, L3_offset, offset)
                fe_prop = torch.cat([fe_t, fe_prop], dim=1)
                fe_prop = self.extract_fe_back(fe_prop)
                img_prop = self.out_from_prop(torch.cat([output[i], fe_prop], 1))
                imgs_from_prop.append(img_prop)
        imgs_from_prop = imgs_from_prop[::-1]
        return torch.stack(imgs_from_prop, dim=1)