from functools import partial

import torch
from modules.sam.sam_image_encoder import ImageEncoderViT


def build_img_encoder(args):
    image_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=args.input_resolution,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        pretrained=args.sam_path,
    )
    added_weight = image_encoder.init_weights()
    if args.fix_img_encoder:
        for name, p in image_encoder.named_parameters():
            if name not in added_weight:  # pretrained weight
                p.requires_grad = False
    return image_encoder
