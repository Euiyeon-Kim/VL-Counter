"""
    Modified from https://github.com/xmed-lab/CLIP_Surgery
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.clip.clip_modules import LayerNorm, QuickGELU, DropPath


# Implement attention module for v-v self-attention
class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # resnets have only one self-attention, norm and larger scale perform better
        if self.settings == 'resnet':
            k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            q = k
            scale = self.scale * 8
        else:
            scale = self.scale

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # clip_surgery
        # x = v.transpose(1, 2).reshape(B, N, C) # mask_clip
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # dual paths for blocks deeper than "d"
        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.drop_path(self.attention(self.ln_1(x_ori)))
                x_res, x_ori_res = x_res
                x_ori = x_ori + x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x = x + x_res  # skip ffn for the new path
                return [x, x_ori]

            # start of dual path
            else:
                x_res = self.drop_path(self.attention(self.ln_1(x)))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x = x + x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
            return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPSurgeryVisionEncoder(nn.Module):
    def __init__(self, input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512,
                 drop_path_rate=0.0, pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.added_weight_names = []

        self.num_heads = heads
        self.embed_dim = width
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.spatial_size = input_resolution // patch_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.attn = None       # Empty attention for v-v attention
        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    spatial_pos = F.interpolate(
                        state_dict["positional_embedding"][1:, ].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
                        size=(self.spatial_size, self.spatial_size), mode='bilinear', align_corners=False)
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size * self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            self.added_weight_names = u
            print(u, w, 'are misaligned params in CLIP vision transformer')
            return u

    def forward(self, x: torch.Tensor):
        # reform the architecture during first inference
        if self.attn == None:
            # apply architecture surgery on the last 6 blocks
            for i in range(1, 7):  # surgery 7, maskclip 2
                self.attn = Attention(self.embed_dim, self.embed_dim, self.num_heads, True)
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn

        x = self.conv1(x)                           # (B, width, grid_h, grid_w)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)   # (B, width, grid_h * grid_w)
        x = x.permute(0, 2, 1)                      # (B, grid_h * grid_w, width)
        x = torch.cat([self.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :]
        spatial_pos = F.interpolate(pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2),
                                    size=(H, W), mode='bilinear', align_corners=False)
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x, x_ori = self.transformer(x)
        x[0, :, :] = x_ori[0, :, :]         # clip_surgery

        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x @ self.proj

        x_ori = x_ori.permute(1, 0, 2)
        x_ori = self.ln_post(x_ori)
        x_ori = x_ori @ self.proj
        x_ori = x_ori[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W

        global_embedding = x[:, 0]
        visual_embedding = x[:, 1:]     # .reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W

        return global_embedding, visual_embedding, x_ori


if __name__ == '__main__':
    clip_img = CLIPSurgeryVisionEncoder(input_resolution=512, pretrained='../../pretrained/ViT-B-16.pt')
    clip_img.init_weights()
    cls, viz = clip_img(torch.rand((2, 3, 512, 512)))
