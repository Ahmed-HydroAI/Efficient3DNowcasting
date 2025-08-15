
# src/efficient3d/models/depthwise_unet3d_legacy.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depthwise U-Net 3D 

"""

import torch
import torch.nn as nn

from blocks_3d import (
    Conv3D_Block,
    DepthwiseConv3D_Block,
    Deconv3D_Block,
    get_norm
)


class DepthwiseUNet3D_3in_1out(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feat_channels=[32, 64, 128, 256, 512, 1024], residual=True):
        super(DepthwiseUNet3D_3in_1out, self).__init__()

        # ====================== Encoder ======================
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))

        # Convolutional Blocks
        self.conv_blk1 = Conv3D_Block(in_channels, feat_channels[0],
                                      norm_type='group', num_groups=8, residual=False)

        self.conv_blk2 = DepthwiseConv3D_Block(feat_channels[0], feat_channels[1],
                                                norm_type='group', num_groups=16, residual=True)

        self.conv_blk3 = DepthwiseConv3D_Block(feat_channels[1], feat_channels[2],
                                                norm_type='group', num_groups=32, residual=True)

        self.conv_blk4 = DepthwiseConv3D_Block(feat_channels[2], feat_channels[3],
                                                norm_type='group', num_groups=32, residual=True)

        # ==================== Bottleneck =====================
        self.bottleneck = Conv3D_Block(feat_channels[3], feat_channels[4],
                                       norm_type='group', num_groups=32, residual=True)

        # ====================== Decoder ======================
        self.deconv_blk4   = Deconv3D_Block(feat_channels[4], feat_channels[3], norm_type='group', num_groups=32)
        self.dec_conv_blk4 = DepthwiseConv3D_Block(2 * feat_channels[3], feat_channels[3],
                                                    norm_type='group', num_groups=32, residual=True)

        self.deconv_blk3   = Deconv3D_Block(feat_channels[3], feat_channels[2], norm_type='group', num_groups=32)
        self.dec_conv_blk3 = DepthwiseConv3D_Block(2 * feat_channels[2], feat_channels[2],
                                                    norm_type='group', num_groups=32, residual=True)

        self.deconv_blk2   = Deconv3D_Block(feat_channels[2], feat_channels[1], norm_type='group', num_groups=16)
        self.dec_conv_blk2 = DepthwiseConv3D_Block(2 * feat_channels[1], feat_channels[1],
                                                    norm_type='group', num_groups=16, residual=True)

        self.deconv_blk1   = Deconv3D_Block(feat_channels[1], feat_channels[0], norm_type='group', num_groups=8)
        self.dec_conv_blk1 = DepthwiseConv3D_Block(2 * feat_channels[0], feat_channels[0],
                                                    norm_type='group', num_groups=8, residual=True)

        # ======================= Output =======================
        self.final_conv = nn.Conv3d(feat_channels[0], out_channels,
                                    kernel_size=(3, 1, 1), stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x):
        # Encoder
        x1 = self.conv_blk1(x)
        x2 = self.conv_blk2(self.pool1(x1))
        x3 = self.conv_blk3(self.pool2(x2))
        x4 = self.conv_blk4(self.pool3(x3))

        # Bottleneck
        base = self.dropout(self.bottleneck(self.pool4(x4)))

        # Decoder
        d4 = self.dec_conv_blk4(torch.cat([self.deconv_blk4(base), x4], dim=1))
        d3 = self.dec_conv_blk3(torch.cat([self.deconv_blk3(d4), x3], dim=1))
        d2 = self.dec_conv_blk2(torch.cat([self.deconv_blk2(d3), x2], dim=1))
        d1 = self.dec_conv_blk1(torch.cat([self.deconv_blk1(d2), x1], dim=1))

        return self.final_conv(d1)


class DepthwiseUNet3D_4in_12out(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, in_frames=4, out_frames=12,
                 feat_channels=[32, 64, 128, 256, 512, 1024], residual=True):
        super(DepthwiseUNet3D_4in_12out, self).__init__()

        self.in_frames = in_frames
        self.out_frames = out_frames

        # ====================== Encoder ======================
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))

        self.conv_blk1 = Conv3D_Block(in_channels, feat_channels[0],
                                      norm_type='group', num_groups=8, residual=False)

        self.conv_blk2 = DepthwiseConv3D_Block(feat_channels[0], feat_channels[1],
                                                norm_type='group', num_groups=16, residual=True)

        self.conv_blk3 = DepthwiseConv3D_Block(feat_channels[1], feat_channels[2],
                                                norm_type='group', num_groups=32, residual=True)

        self.conv_blk4 = DepthwiseConv3D_Block(feat_channels[2], feat_channels[3],
                                                norm_type='group', num_groups=32, residual=True)

        # ==================== Bottleneck =====================
        self.bottleneck = Conv3D_Block(feat_channels[3], feat_channels[4],
                                       norm_type='group', num_groups=32, residual=True)

        # ====================== Decoder ======================
        self.deconv_blk4   = Deconv3D_Block(feat_channels[4], feat_channels[3], norm_type='group', num_groups=32)
        self.dec_conv_blk4 = DepthwiseConv3D_Block(2 * feat_channels[3], feat_channels[3],
                                                    norm_type='group', num_groups=32, residual=True)

        self.deconv_blk3   = Deconv3D_Block(feat_channels[3], feat_channels[2], norm_type='group', num_groups=32)
        self.dec_conv_blk3 = DepthwiseConv3D_Block(2 * feat_channels[2], feat_channels[2],
                                                    norm_type='group', num_groups=32, residual=True)

        self.deconv_blk2   = Deconv3D_Block(feat_channels[2], feat_channels[1], norm_type='group', num_groups=16)
        self.dec_conv_blk2 = DepthwiseConv3D_Block(2 * feat_channels[1], feat_channels[1],
                                                    norm_type='group', num_groups=16, residual=True)

        self.deconv_blk1   = Deconv3D_Block(feat_channels[1], feat_channels[0], norm_type='group', num_groups=8)
        self.dec_conv_blk1 = DepthwiseConv3D_Block(2 * feat_channels[0], feat_channels[0],
                                                    norm_type='group', num_groups=8, residual=True)

        # ======================= Output =======================
        self.expansion_factor = out_frames // in_frames  # 12/4=3
        self.final_conv = nn.Conv3d(
            feat_channels[0],
            out_channels * self.expansion_factor,  # e.g., 3 channels
            kernel_size=(1, 1, 1),
            padding=0,
            bias=True
        )

        self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x):
        # Encoder
        x1 = self.conv_blk1(x)
        x2 = self.conv_blk2(self.pool1(x1))
        x3 = self.conv_blk3(self.pool2(x2))
        x4 = self.conv_blk4(self.pool3(x3))

        # Bottleneck
        base = self.dropout(self.bottleneck(self.pool4(x4)))

        # Decoder
        d4 = self.dec_conv_blk4(torch.cat([self.deconv_blk4(base), x4], dim=1))
        d3 = self.dec_conv_blk3(torch.cat([self.deconv_blk3(d4), x3], dim=1))
        d2 = self.dec_conv_blk2(torch.cat([self.deconv_blk2(d3), x2], dim=1))
        d1 = self.dec_conv_blk1(torch.cat([self.deconv_blk1(d2), x1], dim=1))

        # Output: [B, C_out*factor, T, H, W] -> [B, 1, 12, H, W]
        out = self.final_conv(d1)
        out = out.permute(0, 2, 1, 3, 4)                      # [B, T=4, C=3, H, W]
        out = out.reshape(x.size(0), -1, x.size(3), x.size(4)) # [B, 12, H, W]
        out = out.unsqueeze(1)                                 # [B, 1, 12, H, W]
        return out


class DepthwiseUNet3D_12in_12out(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 feat_channels=[32, 64, 128, 256, 512, 1024], residual=True):
        super(DepthwiseUNet3D_12in_12out, self).__init__()

        # ====================== Encoder ======================
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))

        self.conv_blk1 = Conv3D_Block(in_channels, feat_channels[0],
                                      norm_type='group', num_groups=8, residual=False)

        self.conv_blk2 = DepthwiseConv3D_Block(feat_channels[0], feat_channels[1],
                                                norm_type='group', num_groups=16, residual=True)

        self.conv_blk3 = DepthwiseConv3D_Block(feat_channels[1], feat_channels[2],
                                                norm_type='group', num_groups=32, residual=True)

        self.conv_blk4 = DepthwiseConv3D_Block(feat_channels[2], feat_channels[3],
                                                norm_type='group', num_groups=32, residual=True)

        # ==================== Bottleneck =====================
        self.bottleneck = Conv3D_Block(feat_channels[3], feat_channels[4],
                                       norm_type='group', num_groups=32, residual=True)

        # ====================== Decoder ======================
        self.deconv_blk4   = Deconv3D_Block(feat_channels[4], feat_channels[3], norm_type='group', num_groups=32)
        self.dec_conv_blk4 = DepthwiseConv3D_Block(2 * feat_channels[3], feat_channels[3],
                                                    norm_type='group', num_groups=32, residual=True)

        self.deconv_blk3   = Deconv3D_Block(feat_channels[3], feat_channels[2], norm_type='group', num_groups=32)
        self.dec_conv_blk3 = DepthwiseConv3D_Block(2 * feat_channels[2], feat_channels[2],
                                                    norm_type='group', num_groups=32, residual=True)

        self.deconv_blk2   = Deconv3D_Block(feat_channels[2], feat_channels[1], norm_type='group', num_groups=16)
        self.dec_conv_blk2 = DepthwiseConv3D_Block(2 * feat_channels[1], feat_channels[1],
                                                    norm_type='group', num_groups=16, residual=True)

        self.deconv_blk1   = Deconv3D_Block(feat_channels[1], feat_channels[0], norm_type='group', num_groups=8)
        self.dec_conv_blk1 = DepthwiseConv3D_Block(2 * feat_channels[0], feat_channels[0],
                                                    norm_type='group', num_groups=8, residual=True)

        # ======================= Output =======================
        self.final_conv = nn.Conv3d(feat_channels[0], out_channels,
                                    kernel_size=(1, 1, 1), stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x):
        # Encoder
        x1 = self.conv_blk1(x)
        x2 = self.conv_blk2(self.pool1(x1))
        x3 = self.conv_blk3(self.pool2(x2))
        x4 = self.conv_blk4(self.pool3(x3))

        # Bottleneck
        base = self.dropout(self.bottleneck(self.pool4(x4)))

        # Decoder
        d4 = self.dec_conv_blk4(torch.cat([self.deconv_blk4(base), x4], dim=1))
        d3 = self.dec_conv_blk3(torch.cat([self.deconv_blk3(d4), x3], dim=1))
        d2 = self.dec_conv_blk2(torch.cat([self.deconv_blk2(d3), x2], dim=1))
        d1 = self.dec_conv_blk1(torch.cat([self.deconv_blk1(d2), x1], dim=1))

        return self.final_conv(d1)

