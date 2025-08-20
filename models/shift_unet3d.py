
"""
Shift U-Net 3D 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



from blocks_3d import (
    Conv3D_Block,
    ShiftConv3D,
    Deconv3D_Block,
    get_norm
    
)



class ShiftUNet3D_12in_12out(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feat_channels=[32, 64, 128, 256, 512]):
        super().__init__()
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))
        self.dropout = nn.Dropout3d(p=0.5)

        # Encoder
        self.conv_blk1 = ShiftConv3D(in_channels, feat_channels[0], num_groups=8)
        self.conv_blk2 = ShiftConv3D(feat_channels[0], feat_channels[1], num_groups=16)
        self.conv_blk3 = ShiftConv3D(feat_channels[1], feat_channels[2], num_groups=32)
        self.conv_blk4 = ShiftConv3D(feat_channels[2], feat_channels[3], num_groups=32)

        # Bottleneck
        self.bottleneck = ShiftConv3D(feat_channels[3], feat_channels[4], num_groups=32)

        # Decoder
        self.deconv_blk4   = Deconv3D_Block(feat_channels[4], feat_channels[3], num_groups=32)
        self.dec_conv_blk4 = ShiftConv3D(2 * feat_channels[3], feat_channels[3], num_groups=32)

        self.deconv_blk3   = Deconv3D_Block(feat_channels[3], feat_channels[2], num_groups=32)
        self.dec_conv_blk3 = ShiftConv3D(2 * feat_channels[2], feat_channels[2], num_groups=32)

        self.deconv_blk2   = Deconv3D_Block(feat_channels[2], feat_channels[1], num_groups=16)
        self.dec_conv_blk2 = ShiftConv3D(2 * feat_channels[1], feat_channels[1], num_groups=16)

        self.deconv_blk1   = Deconv3D_Block(feat_channels[1], feat_channels[0], num_groups=8)
        self.dec_conv_blk1 = ShiftConv3D(2 * feat_channels[0], feat_channels[0], num_groups=8)

        self.final_conv = nn.Conv3d(feat_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_blk1(x)
        x2 = self.conv_blk2(self.pool1(x1))
        x3 = self.conv_blk3(self.pool2(x2))
        x4 = self.conv_blk4(self.pool3(x3))
        base = self.dropout(self.bottleneck(self.pool4(x4)))

        d4 = self.dec_conv_blk4(torch.cat([self.deconv_blk4(base), x4], dim=1))
        d3 = self.dec_conv_blk3(torch.cat([self.deconv_blk3(d4), x3], dim=1))
        d2 = self.dec_conv_blk2(torch.cat([self.deconv_blk2(d3), x2], dim=1))
        d1 = self.dec_conv_blk1(torch.cat([self.deconv_blk1(d2), x1], dim=1))

        return self.final_conv(d1)


class ShiftUNet3D_4in_12out(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, in_frames=4, out_frames=12, feat_channels=[32, 64, 128, 256, 512]):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.expansion_factor = out_frames // in_frames

        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))
        self.dropout = nn.Dropout3d(p=0.5)

        # Encoder
        self.conv_blk1 = ShiftConv3D(in_channels, feat_channels[0], num_groups=8)
        self.conv_blk2 = ShiftConv3D(feat_channels[0], feat_channels[1], num_groups=16)
        self.conv_blk3 = ShiftConv3D(feat_channels[1], feat_channels[2], num_groups=32)
        self.conv_blk4 = ShiftConv3D(feat_channels[2], feat_channels[3], num_groups=32)

        # Bottleneck
        self.bottleneck = ShiftConv3D(feat_channels[3], feat_channels[4], num_groups=32)

        # Decoder
        self.deconv_blk4   = Deconv3D_Block(feat_channels[4], feat_channels[3], num_groups=32)
        self.dec_conv_blk4 = ShiftConv3D(2 * feat_channels[3], feat_channels[3], num_groups=32)

        self.deconv_blk3   = Deconv3D_Block(feat_channels[3], feat_channels[2], num_groups=32)
        self.dec_conv_blk3 = ShiftConv3D(2 * feat_channels[2], feat_channels[2], num_groups=32)

        self.deconv_blk2   = Deconv3D_Block(feat_channels[2], feat_channels[1], num_groups=16)
        self.dec_conv_blk2 = ShiftConv3D(2 * feat_channels[1], feat_channels[1], num_groups=16)

        self.deconv_blk1   = Deconv3D_Block(feat_channels[1], feat_channels[0], num_groups=8)
        self.dec_conv_blk1 = ShiftConv3D(2 * feat_channels[0], feat_channels[0], num_groups=8)

        self.final_conv = nn.Conv3d(
            feat_channels[0],
            out_channels * self.expansion_factor,
            kernel_size=(1, 1, 1),
            padding=0,
            bias=True
        )

    def forward(self, x):
        x1 = self.conv_blk1(x)
        x2 = self.conv_blk2(self.pool1(x1))
        x3 = self.conv_blk3(self.pool2(x2))
        x4 = self.conv_blk4(self.pool3(x3))
        base = self.dropout(self.bottleneck(self.pool4(x4)))

        d4 = self.dec_conv_blk4(torch.cat([self.deconv_blk4(base), x4], dim=1))
        d3 = self.dec_conv_blk3(torch.cat([self.deconv_blk3(d4), x3], dim=1))
        d2 = self.dec_conv_blk2(torch.cat([self.deconv_blk2(d3), x2], dim=1))
        d1 = self.dec_conv_blk1(torch.cat([self.deconv_blk1(d2), x1], dim=1))

        # reshape: [B, C_out*factor, T, H, W] -> [B, 1, 12, H, W]
        out = self.final_conv(d1)
        out = out.permute(0, 2, 1, 3, 4)                         # [B, T=4, C=3, H, W]
        out = out.reshape(x.size(0), -1, x.size(3), x.size(4))   # [B, 12, H, W]
        out = out.unsqueeze(1)                                   # [B, 1, 12, H, W]
        return out


# # --------- quick local sanity (optional) ---------
# if __name__ == "__main__":
#     with torch.no_grad():
#         m = Shift_UNet3D_4in_12out()
#         x = torch.randn(1, 1, 4, 64, 64)
#         y = m(x)
#         assert y.shape == (1, 1, 12, 64, 64)
#         print("Shift_UNet3D (legacy) OK")
