# src/efficient3d/models/group_unet3d_legacy.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group U-Net 3D 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F



from blocks_3d import (
    Conv3D_Block,
    GroupConv3D_Block,
    Deconv3D_Block,
)


# ---------- Models (legacy-exact names & heads) ----------

class GroupUNet3D_12in_12out(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feat_channels=[32, 64, 128, 256, 512], conv_groups=4):
        super().__init__()

        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))
        self.dropout = nn.Dropout3d(p=0.5)

        self.conv_blk1 = Conv3D_Block(in_channels, feat_channels[0], norm_type='group', num_groups=8, residual=False)
        self.conv_blk2 = GroupConv3D_Block(feat_channels[0], feat_channels[1], conv_groups=conv_groups, norm_groups=16)
        self.conv_blk3 = GroupConv3D_Block(feat_channels[1], feat_channels[2], conv_groups=conv_groups, norm_groups=32)
        self.conv_blk4 = GroupConv3D_Block(feat_channels[2], feat_channels[3], conv_groups=conv_groups, norm_groups=32)
        self.bottleneck = GroupConv3D_Block(feat_channels[3], feat_channels[4], conv_groups=conv_groups, norm_groups=32)

        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3], norm_type='group', num_groups=32)
        self.dec_conv_blk4 = GroupConv3D_Block(2 * feat_channels[3], feat_channels[3], conv_groups=conv_groups, norm_groups=32)

        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2], norm_type='group', num_groups=32)
        self.dec_conv_blk3 = GroupConv3D_Block(2 * feat_channels[2], feat_channels[2], conv_groups=conv_groups, norm_groups=32)

        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1], norm_type='group', num_groups=16)
        self.dec_conv_blk2 = GroupConv3D_Block(2 * feat_channels[1], feat_channels[1], conv_groups=conv_groups, norm_groups=16)

        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0], norm_type='group', num_groups=8)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], norm_type='group', num_groups=8, residual=True)

        self.final_conv = nn.Conv3d(feat_channels[0], out_channels, kernel_size=1)

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



class GroupUNet3D_4in_12out(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, in_frames=4, out_frames=12,
                 feat_channels=[32, 64, 128, 256, 512], conv_groups=4):
        super().__init__()

        self.in_frames = in_frames
        self.out_frames = out_frames
        self.expansion_factor = out_frames // in_frames  # 12 // 4 = 3

        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))
        self.dropout = nn.Dropout3d(p=0.5)

        self.conv_blk1 = Conv3D_Block(in_channels, feat_channels[0], norm_type='group', num_groups=8, residual=False)
        self.conv_blk2 = GroupConv3D_Block(feat_channels[0], feat_channels[1], conv_groups=conv_groups, norm_groups=16)
        self.conv_blk3 = GroupConv3D_Block(feat_channels[1], feat_channels[2], conv_groups=conv_groups, norm_groups=32)
        self.conv_blk4 = GroupConv3D_Block(feat_channels[2], feat_channels[3], conv_groups=conv_groups, norm_groups=32)
        self.bottleneck = GroupConv3D_Block(feat_channels[3], feat_channels[4], conv_groups=conv_groups, norm_groups=32)

        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3], norm_type='group', num_groups=32)
        self.dec_conv_blk4 = GroupConv3D_Block(2 * feat_channels[3], feat_channels[3], conv_groups=conv_groups, norm_groups=32)

        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2], norm_type='group', num_groups=32)
        self.dec_conv_blk3 = GroupConv3D_Block(2 * feat_channels[2], feat_channels[2], conv_groups=conv_groups, norm_groups=32)

        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1], norm_type='group', num_groups=16)
        self.dec_conv_blk2 = GroupConv3D_Block(2 * feat_channels[1], feat_channels[1], conv_groups=conv_groups, norm_groups=16)

        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0], norm_type='group', num_groups=8)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], norm_type='group', num_groups=8, residual=True)

        self.final_conv = nn.Conv3d(
            feat_channels[0],
            out_channels * self.expansion_factor,
            kernel_size=1
        )

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

        # Final prediction and reshape
        out = self.final_conv(d1)                                   # [B, C_out*factor, T, H, W]
        out = out.permute(0, 2, 1, 3, 4)                            # [B, T=4, C=3, H, W]
        out = out.reshape(x.size(0), self.out_frames, x.size(3), x.size(4))  # [B, 12, H, W]
        return out.unsqueeze(1)                                     # [B, 1, 12, H, W]


# # Optional camelCase aliases if any scripts import them:
# class GroupUNet3D_12in_12out(Group_UNet3D_12in_12out):
#     pass

# class GroupUNet3D_4in_12out(Group_UNet3D_4in_12out):
#     pass

