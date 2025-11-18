import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from itertools import repeat
import collections.abc
import numpy as np



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

class PatchMerging_noFAA(nn.Module):

    def __init__(self, input_resolution, dim, block, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction1 = nn.Linear(64, 32)
        self.reduction2 = nn.Linear(640, 320)
        self.reduction3 = nn.Linear(32, 16)
        self.reduction4 = nn.Linear(320, 160)
        self.norm1 = norm_layer(64)
        self.norm2 = norm_layer(32)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, H, W = x.shape#64 64 640
        if H==64:
            x = x.permute(0, 2, 1)
            x = self.norm1(x)
            x = self.reduction1(x)
            x = x.permute(0, 2, 1)
            x = self.reduction2(x)
        elif H==32:
            x = x.permute(0, 2, 1)
            x = self.norm2(x)
            x = self.reduction3(x)
            x = x.permute(0, 2, 1)
            x = self.reduction4(x)

        return x.squeeze()

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    # 这个地方有两个窗口, 一个是patch内像素分割的窗口, 另一个是块间的窗口，用于计算注意力。

    def __init__(self, input_resolution, dim, block, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.block = block
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.window_size = 4
        self.patch_window = 4
        self.num_heads = 4
        # dim 表示的是输入通道的维度?????????  在本工作中,我们考虑的应该是当前窗口内的所有特征点.
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads, qkv_bias=True, qk_scale=True, attn_drop=0.1, proj_drop=0.2)
        # 关于这个窗口的重点内容应该聚焦于如何探讨窗口内的useful windows
        self.Unfold = nn.Unfold(kernel_size=(self.patch_window, self.patch_window), stride=(self.patch_window, self.patch_window))
        self.Fold = nn.Fold(output_size=(input_resolution[0] // 2, input_resolution[1] // 2), kernel_size=(self.patch_window, self.patch_window), stride=(self.patch_window, self.patch_window))

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, _, _ = x.shape
        # 分块
        x = x.unsqueeze(1)  # B C Freq  Frame  where C=1 64 1 64 640
        x = self.Unfold(x).contiguous()  # 输出格式：batch, c×k1×k2, patch块数量 64 1*4*4 2560
        x = x.reshape(B, -1, self.input_resolution[0] // self.patch_window, self.input_resolution[1] // self.patch_window)#64 16 16 160
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, F = x.shape# 64 16 160 16
        # -------------------------------------------------
        "块"
        x = window_partition(x, self.window_size) # 10240 4 4 16
        x = x.view(-1, self.window_size * self.window_size, F)  # nW*B, window_size*window_size, C 10240 16 16
        # 注意力机制针对的是分块后的每个块 就是哪一部分的块（patch-->frequency and frame）更重要。
        x, attn = self.attn(x)
        # merge windows
        x = x.view(-1, self.window_size, self.window_size, F)#10240 4 4 16
        x = window_reverse(x, self.window_size, H, W)#64 16 160 16
        # -------------------------------------------------
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        # 所有行
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C 64 8 80 16
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # 所有列
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        """
        目标： 选择🈶的特征进行保留。
        按照块去选择，上下左右四个块 根据网络选择其中useful的特征
        在特征中进行拼接，
        按照汉和列单独选择
        """
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*F  # 特征维度的拼接 64 8 80 64
        x = self.norm(x)
        x = self.reduction(x)  # B, H//2, W//2, F 64 8 80 16
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, self.patch_window ** 2, -1).contiguous()#64 16 640
        x = self.Fold(x)#64 32 320
        return x.squeeze()

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


# 按照窗口分区
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  # 坐标coordinates
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='xy'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch-script happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        """位置编码作用提升2-3%"""
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CSWinBlock(nn.Module):

    def __init__(self, reso, num_heads,dim,
                 split_size, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.norm1 = norm_layer(dim)

        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    self.dim, input_resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    self.dim // 2, input_resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop)
                for i in range(self.branch_num)])

        # self.stage1_conv_embed = nn.Sequential(
        #     nn.Conv2d(1, 1, 7, 4, 2),
        #     Rearrange('b c h w -> b (h w) c', h=self.patches_resolution[0] // 4,w = self.patches_resolution[1] // 4),
        #     # nn.LayerNorm(embed_dim)
        # )


    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = x.unsqueeze(1)
        batch, C, H, W = x.shape

        x = torch.reshape(x, (batch, -1, H//4, 4, W//4, 4))
        x = torch.reshape(x, (batch, H//4*W//4, 16*C)).contiguous()
        B,_,C = x.shape

        # img = self.norm1(x)
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=3)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)

        "这里需要对attened_x处理"
        x = x + self.norm1(attened_x)
        x = x.transpose(-2, -1).contiguous().view(B, self.num_heads, H, W)
        x = torch.mean(x,dim = 1)

        return x

class LePEAttention(nn.Module):
    def __init__(self, dim, input_resolution,idx, split_size, num_heads, attn_drop=0.,qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out =  dim
        self.H = input_resolution[0]//4
        self.W = input_resolution[1]//4
        self.split_size = split_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = 1, self.split_size
        elif idx == 1:
            W_sp, H_sp = 1, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)

        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)


    def im2cswin(self, x):
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        H = self.H
        W = self.W
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        H = self.H
        W = self.W
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        # img =


        q, k, v = qkv[0], qkv[1], qkv[2]
        ### Img2Window
        B, L, C = q.shape
        # assert L == self.H * self.W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, self.H, self.W).view(B, -1, C)  # B H' W' C
        # x = windows2img(x, self.H_sp, self.W_sp, self.H, self.W)

        return x.squeeze()

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x