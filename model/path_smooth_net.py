import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class PathSmoothUNet(nn.Module):

    def __init__(self, in_chn, wf=32, depth=4, relu_slope=0.2):
        super(PathSmoothUNet, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = self.get_input_chn(in_chn)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        subnet_repeat_num = 1
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope, subnet_repeat_num))
            prev_channels = (2**i)*wf
            subnet_repeat_num += 1

        self.last = conv3x3(prev_channels, 2, bias=True)

    def forward(self, x1):
        blocks = []
        for i, down in enumerate(self.down_path):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)
        for i, up in enumerate(self.up_path):
            x1 = up(x1, blocks[-i-1])

        pred = self.last(x1)
        return pred

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True))

        self.downsample = downsample
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class BinearUp(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=2):
        super(BinearUp, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale_factor = scale_factor

        self.decoder = nn.Sequential(
            nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, stride=1, padding=1,
                               bias=True),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

        diffY = torch.tensor([skip.size()[2] - x.size()[2]])
        diffX = torch.tensor([skip.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        return self.decoder(x)


class CAConvBlock(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope, subnet_repeat_num):
        super(UNetUpBlock, self).__init__()
        self.up = BinearUp(in_size, out_size)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)
        self.skip_m = skip_blocks(out_size, out_size, subnet_repeat_num)
        self.caconv = CAConvBlock(in_size)

    def forward(self, x, bridge):
        up = self.up(x, bridge)
        bridge = self.skip_m(bridge)
        out = torch.cat([up, bridge], 1)
        out = self.caconv(out)
        out = self.conv_block(out)
        return out


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc
