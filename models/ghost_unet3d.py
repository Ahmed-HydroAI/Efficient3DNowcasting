# src/efficient3d/models/ghost_unet3d_legacy.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ghost U-Net 3D (legacy-exact)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F




from blocks_3d import (
    Conv3D_Block,
    GhostConv3D_Block,
    Deconv3D_Block,
    get_norm
)




# ---- Model (legacy-exact names & head) ----

class GhostUNet3D_4in_12out(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, in_frames=4, out_frames=12,
                 feat_channels=[32, 64, 128, 256, 512, 1024], residual=True):
        super().__init__()

        self.in_frames = in_frames
        self.out_frames = out_frames
        self.expansion_factor = out_frames // in_frames

        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))

        self.conv_blk1 = GhostConv3D_Block(in_channels,           feat_channels[0], norm_type='group', num_groups=8,  residual=False)
        self.conv_blk2 = GhostConv3D_Block(feat_channels[0],      feat_channels[1], norm_type='group', num_groups=16, residual=True)
        self.conv_blk3 = GhostConv3D_Block(feat_channels[1],      feat_channels[2], norm_type='group', num_groups=32, residual=True)
        self.conv_blk4 = GhostConv3D_Block(feat_channels[2],      feat_channels[3], norm_type='group', num_groups=32, residual=True)

        self.bottleneck = GhostConv3D_Block(feat_channels[3],     feat_channels[4], norm_type='group', num_groups=32, residual=True)
        self.dropout_bottleneck = nn.Dropout3d(p=0.5)

        self.deconv_blk4   = Deconv3D_Block(feat_channels[4], feat_channels[3], norm_type='group', num_groups=32)
        self.dec_conv_blk4 = GhostConv3D_Block(2 * feat_channels[3], feat_channels[3], norm_type='group', num_groups=32, residual=True)

        self.deconv_blk3   = Deconv3D_Block(feat_channels[3], feat_channels[2], norm_type='group', num_groups=32)
        self.dec_conv_blk3 = GhostConv3D_Block(2 * feat_channels[2], feat_channels[2], norm_type='group', num_groups=32, residual=True)
        self.dropout_dec3  = nn.Dropout3d(p=0.3)

        self.deconv_blk2   = Deconv3D_Block(feat_channels[2], feat_channels[1], norm_type='group', num_groups=16)
        self.dec_conv_blk2 = GhostConv3D_Block(2 * feat_channels[1], feat_channels[1], norm_type='group', num_groups=16, residual=True)

        self.deconv_blk1   = Deconv3D_Block(feat_channels[1], feat_channels[0], norm_type='group', num_groups=8)
        self.dec_conv_blk1 = GhostConv3D_Block(2 * feat_channels[0], feat_channels[0], norm_type='group', num_groups=8, residual=True)

        self.final_conv = nn.Conv3d(
            feat_channels[0],
            out_channels * self.expansion_factor,
            kernel_size=1,
            padding=0,
            bias=True
        )

    def forward(self, x):
        x1 = self.conv_blk1(x)
        x2 = self.conv_blk2(self.pool1(x1))
        x3 = self.conv_blk3(self.pool2(x2))
        x4 = self.conv_blk4(self.pool3(x3))

        base = self.dropout_bottleneck(self.bottleneck(self.pool4(x4)))

        d4 = self.dec_conv_blk4(torch.cat([self.deconv_blk4(base), x4], dim=1))
        d3_in = torch.cat([self.deconv_blk3(d4), x3], dim=1)
        d3 = self.dropout_dec3(self.dec_conv_blk3(d3_in))
        d2 = self.dec_conv_blk2(torch.cat([self.deconv_blk2(d3), x2], dim=1))
        d1 = self.dec_conv_blk1(torch.cat([self.deconv_blk1(d2), x1], dim=1))

        out = self.final_conv(d1)
        out = out.permute(0, 2, 1, 3, 4)
        out = out.reshape(x.size(0), self.out_frames, x.size(3), x.size(4))
        return out.unsqueeze(1)


# # Optional alias (if some scripts import camelCase)
# class GhostUNet3D_4in_12out(Ghost_UNet3D_4in_12out):
#     pass


# # ---- quick sanity (optional) ----
# if __name__ == "__main__":
#     with torch.no_grad():
#         m = Ghost_UNet3D_4in_12out()
#         x = torch.randn(1, 1, 4, 64, 64)
#         y = m(x)
#         assert y.shape == (1, 1, 12, 64, 64)
#         print("Ghost_UNet3D_4in_12out (legacy) OK")
# from hybrid_2D_3D import *

