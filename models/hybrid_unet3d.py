
# src/efficient3d/models/hybrid_unet3d_legacy.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks_3d import (
    Conv3D_Block,
    Conv2D_Block,
    Deconv3D_Block,
    get_norm
    
)



# ----  Hybrid UNet (4->12) head  ----

class HybridUNet3D_4in_12out(nn.Module):
    """
    Matches the original architecture and state_dict keys exactly.
    """
    def __init__(self, in_channels=1, out_channels=1, in_frames=4, out_frames=12,
                 feat_channels=(32, 64, 128, 256, 512)):
        super().__init__()

        self.expansion_factor = out_frames // in_frames  # e.g., 12/4 = 3

        # spatial-only pooling
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))

        # Encoder (identical residual flags)
        self.conv_blk1 = Conv2D_Block(in_channels,      feat_channels[0], norm_type='group', num_groups=8,  residual=False)
        self.conv_blk2 = Conv2D_Block(feat_channels[0], feat_channels[1], norm_type='group', num_groups=16, residual=True)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], norm_type='group', num_groups=32, residual=True)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], norm_type='group', num_groups=32, residual=True)

        # Bottleneck
        self.bottleneck = Conv3D_Block(feat_channels[3], feat_channels[4], norm_type='group', num_groups=32, residual=True)
        self.dropout_bottleneck = nn.Dropout3d(p=0.5)

        # Decoder (transpose up + conv) with original names
        self.deconv_blk4   = Deconv3D_Block(feat_channels[4], feat_channels[3], norm_type='group', num_groups=32)
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], norm_type='group', num_groups=32, residual=True)

        self.deconv_blk3   = Deconv3D_Block(feat_channels[3], feat_channels[2], norm_type='group', num_groups=32)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], norm_type='group', num_groups=32, residual=True)
        self.dropout_dec3  = nn.Dropout3d(p=0.3)

        self.deconv_blk2   = Deconv3D_Block(feat_channels[2], feat_channels[1], norm_type='group', num_groups=16)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], norm_type='group', num_groups=16, residual=True)

        self.deconv_blk1   = Deconv3D_Block(feat_channels[1], feat_channels[0], norm_type='group', num_groups=8)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], norm_type='group', num_groups=8, residual=True)

        # Output head (channel->time expansion) with original name
        self.final_conv = nn.Conv3d(
            feat_channels[0],
            out_channels * self.expansion_factor,
            kernel_size=(1, 1, 1),
            padding=0,
            bias=True
        )

    def forward(self, x):
        # Encoder
        x1 = self.conv_blk1(x)
        x2 = self.conv_blk2(self.pool1(x1))
        x3 = self.conv_blk3(self.pool2(x2))
        x4 = self.conv_blk4(self.pool3(x3))

        # Bottleneck
        base = self.dropout_bottleneck(self.bottleneck(self.pool4(x4)))

        # Decoder
        d4 = self.dec_conv_blk4(torch.cat([self.deconv_blk4(base), x4], dim=1))
        d3 = self.dropout_dec3(self.dec_conv_blk3(torch.cat([self.deconv_blk3(d4), x3], dim=1)))
        d2 = self.dec_conv_blk2(torch.cat([self.deconv_blk2(d3), x2], dim=1))
        d1 = self.dec_conv_blk1(torch.cat([self.deconv_blk1(d2), x1], dim=1))

        # Output reshape: [B, C_out*factor, T, H, W] -> [B, 1, 12, H, W]
        out = self.final_conv(d1)
        out = out.permute(0, 2, 1, 3, 4)                       # [B, T=4, C=3, H, W]
        out = out.reshape(x.size(0), -1, x.size(3), x.size(4))  # [B, 12, H, W]
        return out.unsqueeze(1)                                 # [B, 1, 12, H, W]



# class HybridUNet3D_4in_12out(Hybrid_UNet3D_4in_12out):
#     pass


# # ---- quick shape sanity (optional) ----
# if __name__ == "__main__":
#     with torch.no_grad():
#         m = HybridUNet3D_4in_12out()
#         x = torch.randn(2, 1, 4, 64, 64)
#         y = m(x)
#         assert y.shape == (2, 1, 12, 64, 64)
#         print("Hybrid_UNet3D_4in_12out (legacy) OK")

