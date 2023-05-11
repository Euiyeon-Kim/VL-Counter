"""
    Originated from https://github.com/raoyongming/DenseCLIP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.clip.clip_modules import LayerNorm, Transformer


class CLIPVisionEncoder(nn.Module):
    def __init__(self, input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512,
                 drop_path_rate=0.0, out_indices=[3, 5, 7, 11], pretrained=None, get_embeddings=True, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.added_weight_names = []
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices if out_indices else []

        if get_embeddings:
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
            print(u, w, 'are misaligned params in vision transformer')
            return u

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] 
        spatial_pos = F.interpolate(pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2),
                                    size=(H, W), mode='bilinear', align_corners=False)
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(xp.contiguous())

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, 1:] # .reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W

            features.append([global_embedding, visual_embedding])

        return features


if __name__ == '__main__':
    clip_img = CLIPVisionEncoder(pretrained='../../pretrained/ViT-B-16.pt')
    clip_img.init_weights()
