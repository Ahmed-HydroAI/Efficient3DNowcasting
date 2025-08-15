

import torch
import torch.nn as nn
import torch.nn.functional as F



class GroupNormFP32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).to(dtype=x.dtype)
    
    
def get_norm(norm_type, num_channels, num_groups=8):
    if norm_type is None:
        return nn.Identity()
    elif norm_type == 'batch':
        return nn.BatchNorm3d(num_channels)
    elif norm_type == 'instance':
        return nn.InstanceNorm3d(num_channels)
    elif norm_type == 'group':
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        return GroupNormFP32(num_groups=num_groups, num_channels=num_channels)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

class Deconv3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='group', num_groups=8, conv_bias=False):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=conv_bias),
            get_norm(norm_type, out_channels, num_groups),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.deconv(x)

class Upsample3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False):
        super().__init__()

        def get_norm(c):
            return nn.BatchNorm3d(c) if use_bn else GroupNormFP32(4, c)

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
        )

    def forward(self, x):
        return self.block(x)
    
    
class Conv3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='group', num_groups=8, residual=True, conv_bias=False):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias)
        self.norm1 = get_norm(norm_type, out_channels, num_groups)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias)
        self.norm2 = get_norm(norm_type, out_channels, num_groups)

        if residual and in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=conv_bias),
                get_norm(norm_type, out_channels, num_groups)
            )
        elif residual:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.residual:
            out += self.residual_conv(identity)

        return torch.relu(out)


class Conv2D_Block(nn.Module):
    """
    Time-distributed 2D conv block operating per-frame.
    Preserves original attributes: conv1, norm1, conv2, norm2, residual_conv
    Supports inputs of shape [B, C, T, H, W] or [B, C, H, W] (T=1).
    """
    def __init__(self, in_channels, out_channels, norm_type='group', num_groups=8, residual=True):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = get_norm(norm_type, out_channels, num_groups)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, out_channels, num_groups)

        if residual and in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm(norm_type, out_channels, num_groups)
            )
        elif residual:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = None

    def forward(self, x):
        """
        x: [B, C, T, H, W]  or  [B, C, H, W]
        returns: same rank as input (adds/squeezes T=1 when necessary)
        """
        added_time_dim = False

        if x.ndim == 4:
            # Treat as a single-frame sequence: add T=1
            x = x.unsqueeze(2)  # -> [B, C, 1, H, W]
            added_time_dim = True
        elif x.ndim != 5:
            raise ValueError(f"Conv2D_Block expects 4D or 5D input, got {x.ndim}D with shape {tuple(x.shape)}")

        B, C, T, H, W = x.shape

        # Flatten time to batch and run 2D convs per-frame
        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        identity = x2d

        out = F.relu(self.norm1(self.conv1(x2d)))
        out = self.norm2(self.conv2(out))

        if self.residual:
            if isinstance(self.residual_conv, nn.Identity):
                out = out + identity
            elif self.residual_conv is not None:
                out = out + self.residual_conv(identity)

        out = F.relu(out)

        # Restore [B, C_out, T, H, W]
        out = out.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)

        # If input was 4D, return 4D by squeezing T
        if added_time_dim:
            out = out.squeeze(2)  # -> [B, C_out, H, W]

        return out


class DepthwiseConv3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='group', num_groups=8, residual=True, conv_bias=False):
        super().__init__()
        self.residual = residual

        self.depthwise1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=conv_bias)
        self.pointwise1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=conv_bias)
        self.norm1 = get_norm(norm_type, out_channels, num_groups)

        self.depthwise2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=conv_bias)
        self.pointwise2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=conv_bias)
        self.norm2 = get_norm(norm_type, out_channels, num_groups)

        if residual and in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=conv_bias),
                get_norm(norm_type, out_channels, num_groups)
            )
        elif residual:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x

        out = self.depthwise1(x)
        out = self.pointwise1(out)
        out = self.norm1(out)
        out = torch.relu(out)

        out = self.depthwise2(out)
        out = self.pointwise2(out)
        out = self.norm2(out)

        if self.residual:
            out += self.residual_conv(identity)

        return torch.relu(out)
    
class GenericShift3D(nn.Module):
    """3D version of shift operation using channel grouping"""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.num_shifts = kernel_size ** 3  # e.g., 27 for 3x3x3

    def forward(self, x):
        B, C, D, H, W = x.shape
        channels_per_shift = C // self.num_shifts
        remaining = C % self.num_shifts

        # pad order: (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad))
        
        out = []
        for i in range(self.num_shifts):
            dz = i // (self.kernel_size ** 2)
            dy = (i // self.kernel_size) % self.kernel_size
            dx = i % self.kernel_size
            offset = (dz - self.pad, dy - self.pad, dx - self.pad)

            shifted = x_pad[:, i*channels_per_shift:(i+1)*channels_per_shift,
                            self.pad+offset[0]:self.pad+offset[0]+D,
                            self.pad+offset[1]:self.pad+offset[1]+H,
                            self.pad+offset[2]:self.pad+offset[2]+W]
            out.append(shifted)

        if remaining > 0:
            out.append(x_pad[:, -remaining:, self.pad:-self.pad, self.pad:-self.pad, self.pad:-self.pad])

        return torch.cat(out, dim=1)


class ShiftConv3D(nn.Module):
    """3D Shift Block with GroupNorm """
    def __init__(self, in_planes, out_planes, stride=1, expansion=1, num_groups=8):
        super().__init__()
        self.expansion = expansion
        mid_planes = int(out_planes * expansion)
        
        self.conv1 = nn.Conv3d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, mid_planes)
        
        self.shift = GenericShift3D(kernel_size=3)
        
        self.conv2 = nn.Conv3d(mid_planes, out_planes, kernel_size=1,
                               stride=(1, stride, stride), bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.shift(x)
        x = F.relu(self.gn2(self.conv2(x)))
        return x + identity


class R2Plus1D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='group', num_groups=8, residual=True):
        super().__init__()
        self.residual = residual

        # Spatial conv: 1x3x3
        self.conv_spatial = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.norm_spatial = get_norm(norm_type, out_channels, num_groups)

        # Temporal conv: 3x1x1
        self.conv_temporal = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.norm_temporal = get_norm(norm_type, out_channels, num_groups)

        if residual and in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm(norm_type, out_channels, num_groups)
            )
        elif residual:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x
        x = F.relu(self.norm_spatial(self.conv_spatial(x)))
        x = F.relu(self.norm_temporal(self.conv_temporal(x)))
        if self.residual:
            x += self.residual_conv(identity)
        return F.relu(x)
    

class GhostConv3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, norm_type='group', num_groups=8, residual=True):
        super().__init__()
        self.residual = residual
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        ghost_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, init_channels, num_groups),
            nn.ReLU(inplace=True)
        )

        self.cheap_op = nn.Sequential(
            nn.Conv3d(init_channels, ghost_channels, kernel_size=3, padding=1, groups=init_channels, bias=False),
            get_norm(norm_type, ghost_channels, num_groups),
            nn.ReLU(inplace=True)
        )

        if residual and in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm(norm_type, out_channels, num_groups)
            )
        elif residual:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x
        x1 = self.primary_conv(x)
        x2 = self.cheap_op(x1)
        out = torch.cat([x1, x2], dim=1)
        if self.residual:
            if isinstance(self.residual_conv, nn.Identity):
                out = out + identity
            elif self.residual_conv is not None:
                out = out + self.residual_conv(identity)
        return F.relu(out)
    
class GroupConv3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, conv_groups=4, norm_groups=8, residual=True):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, groups=conv_groups, bias=False
        )
        self.norm1 = get_norm("group", out_channels, num_groups=norm_groups)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=conv_groups, bias=False
        )
        self.norm2 = get_norm("group", out_channels, num_groups=norm_groups)

        if residual and (in_channels != out_channels):
            self.residual_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm("group", out_channels, num_groups=norm_groups)
            )
        elif residual:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.residual:
            if isinstance(self.residual_conv, nn.Identity):
                x = x + identity
            elif self.residual_conv is not None:
                x = x + self.residual_conv(identity)
        return F.relu(x)
