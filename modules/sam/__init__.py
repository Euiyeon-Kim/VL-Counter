from functools import partial

import torch
from modules.sam.sam_image_encoder import ImageEncoderViT


def build_img_encoder(args):
    image_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=args.image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=args.prompt_embed_dim,
    )
    return image_encoder


if __name__ == '__main__':
    from dotmap import DotMap
    args = DotMap({
        'image_size': 1024,
        'prompt_embed_dim': 256,
    })

    print()
    print(args.prompt_embed_dim)
    exit()