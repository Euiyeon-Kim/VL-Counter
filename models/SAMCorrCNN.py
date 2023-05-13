from collections import defaultdict

import torch
import torch.nn as nn

from .BaseModel import BaseModel
from modules.decoder import DensityX16
from modules.clip import build_txt_encoder, embed_classname
from modules.sam import build_img_encoder
from modules.sam.sam_modules import LayerNorm2d
from utils.visualize import denormalize, save_density_map_w_similarity


class SAMCorrCNNv1(BaseModel):
    """
        Only ussing correlation
    """
    def __init__(self, args):
        super(SAMCorrCNNv1, self).__init__(args=args)
        self.txt_backbone = build_txt_encoder(args)
        self.img_backbone = build_img_encoder(args)
        self.enhancer = nn.Sequential(
            nn.Conv2d(args.img_emb_dim, args.txt_emb_dim, kernel_size=1, bias=False),
            LayerNorm2d(args.txt_emb_dim),
            nn.Conv2d(args.txt_emb_dim, args.txt_emb_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(args.txt_emb_dim),
        )
        self.decoder = DensityX16(in_dim=args.txt_emb_dim)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def get_log_dict(self):
        denormed = denormalize(self.img_dict['img'].unsqueeze(0), self.args.img_norm_mean, self.args.img_norm_var)
        img = save_density_map_w_similarity(denormed, self.img_dict['pred'], self.img_dict['sim'],
                                            self.img_dict['origin_sim'], self.img_dict['gt'], self.img_dict['count'],
                                            self.img_dict['class_name'], self.args.density_scale)
        self.reset_img_dict()
        return {'pred': torch.from_numpy(img).permute(2, 0, 1) / 255.}

    def inference(self, img, class_name, set_img_dict=False):
        density = self.predict(img, class_name, set_img_dict=set_img_dict)
        return density

    def predict(self, imgs, class_names, set_img_dict=False):
        # Encode txt
        txt_embeddings = embed_classname(self.txt_backbone, class_names)

        # Encode image
        features = self.img_backbone(imgs)
        align_features = self.enhancer(features)
        
        B, _, npH, npW = align_features.shape
        align_features = align_features.reshape(B, -1, npH * npW).permute(0, 2, 1).contiguous()
        
        corr = torch.mul(txt_embeddings.unsqueeze(1), align_features)
        corr = corr.reshape(B, npH, npW, -1).permute(0, 3, 1, 2)
        pred_density = self.decoder(corr)
        
        # Save intermediate results for logging
        if set_img_dict:
            similarity = torch.sum(corr, dim=1, keepdim=True)
            self.img_dict['img'] = imgs[0]
            self.img_dict['pred'] = pred_density[0]
            self.img_dict['sim'] = similarity[0]

        return pred_density

    def forward(self, inp_dict, set_img_dict=False):
        img, class_name = inp_dict['img'].cuda(), inp_dict['class_name']
        pred_density_map = self.predict(img, class_name, set_img_dict)

        # Calculate loss
        metric_dict = {}
        gt = inp_dict['gt']

        l2_density_loss = self.mse_loss(gt, pred_density_map)
        total_loss = l2_density_loss

        metric_dict.update({
            'l2_density': l2_density_loss.item(),
            'total_loss': total_loss.item(),
        })

        if set_img_dict:
            self.img_dict['class_name'] = inp_dict['class_name'][0]
            self.img_dict['gt'] = gt[0]
            self.img_dict['count'] = inp_dict['count'][0]
            
        return total_loss, metric_dict