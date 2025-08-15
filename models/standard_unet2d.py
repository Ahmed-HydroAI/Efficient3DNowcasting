
"""
standard_unet2d

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from blocks_3d import (
    Conv2D_Block,
    Deconv2D_Block,
    get_norm
)


class Standard_UNet2D (nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feat_channels=[32, 64, 128, 256, 512], residual=True):
        super(Standard_UNet2D, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv_blk1 = Conv2D_Block(in_channels, feat_channels[0], norm_type='group', num_groups=8, residual=False)
        self.conv_blk2 = Conv2D_Block(feat_channels[0], feat_channels[1], norm_type='group', num_groups=16, residual=True)
        self.conv_blk3 = Conv2D_Block(feat_channels[1], feat_channels[2], norm_type='group', num_groups=32, residual=True)
        self.conv_blk4 = Conv2D_Block(feat_channels[2], feat_channels[3], norm_type='group', num_groups=32, residual=True)

        self.bottleneck = Conv2D_Block(feat_channels[3], feat_channels[4], norm_type='group', num_groups=32, residual=True)
        self.dropout_bottleneck = nn.Dropout2d(p=0.5)

        self.deconv_blk4 = Deconv2D_Block(feat_channels[4], feat_channels[3], norm_type='group', num_groups=32)
        self.dec_conv_blk4 = Conv2D_Block(2 * feat_channels[3], feat_channels[3], norm_type='group', num_groups=32, residual=True)

        self.deconv_blk3 = Deconv2D_Block(feat_channels[3], feat_channels[2], norm_type='group', num_groups=32)
        self.dec_conv_blk3 = Conv2D_Block(2 * feat_channels[2], feat_channels[2], norm_type='group', num_groups=32, residual=True)
        self.dropout_dec3 = nn.Dropout2d(p=0.3)

        self.deconv_blk2 = Deconv2D_Block(feat_channels[2], feat_channels[1], norm_type='group', num_groups=16)
        self.dec_conv_blk2 = Conv2D_Block(2 * feat_channels[1], feat_channels[1], norm_type='group', num_groups=16, residual=True)

        self.deconv_blk1 = Deconv2D_Block(feat_channels[1], feat_channels[0], norm_type='group', num_groups=8)
        self.dec_conv_blk1 = Conv2D_Block(2 * feat_channels[0], feat_channels[0], norm_type='group', num_groups=8, residual=True)

        self.final_conv = nn.Conv2d(feat_channels[0], out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.conv_blk1(x)
        x2 = self.conv_blk2(self.pool(x1))
        x3 = self.conv_blk3(self.pool(x2))
        x4 = self.conv_blk4(self.pool(x3))

        base = self.dropout_bottleneck(self.bottleneck(self.pool(x4)))

        d4 = self.dec_conv_blk4(torch.cat([self.deconv_blk4(base), x4], dim=1))
        d3 = self.dropout_dec3(self.dec_conv_blk3(torch.cat([self.deconv_blk3(d4), x3], dim=1)))
        d2 = self.dec_conv_blk2(torch.cat([self.deconv_blk2(d3), x2], dim=1))
        d1 = self.dec_conv_blk1(torch.cat([self.deconv_blk1(d2), x1], dim=1))

        return self.final_conv(d1)

# if __name__ == "__main__":
#     model = Standard_UNet2D(in_channels=3, out_channels=1)
#     x = torch.randn(1, 3, 512, 512)  # 3 input frames stacked as channels
#     out = model(x)
#     print(out.shape)  # [1, 1, 256, 256]
