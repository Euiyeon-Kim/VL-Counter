import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel
from modules.decoder import DensityX16, CNNDecoderwCNNFeat
from modules.clip import build_txt_encoder, embed_classname
from modules.clip import build_s_img_encoder as build_img_encoder
from utils.visualize import denormalize, save_density_map_w_similarity


class CLIPSCorrCNNv1(BaseModel):
    """
        Use surgery ViT feature
    """
    def __init__(self, args):
        super(CLIPSCorrCNNv1, self).__init__(args=args)

        self.txt_backbone = build_txt_encoder(args)
        self.img_backbone = build_img_encoder(args)

        self.decoder = DensityX16(in_dim=args.txt_emb_dim * 2)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.simmap_ce_lambda = args.simmap_ce_lambda

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
        B, _, H, W = imgs.size()
        npH, npW = H // self.args.patch_size, W // self.args.patch_size
        _, surg_features, _ = self.img_backbone(imgs)      # [B, 1, 512], [B, 1024, 512]

        # Normalization
        txt_embeddings = txt_embeddings / txt_embeddings.norm(dim=-1, keepdim=True)
        surg_features = surg_features / surg_features.norm(dim=-1, keepdim=True)

        corr = torch.mul(txt_embeddings.unsqueeze(1), surg_features)
        corr = corr.reshape(B, npH, npW, -1).permute(0, 3, 1, 2)
        
        surg_features = surg_features.reshape(B, npH, npW, -1).permute(0, 3, 1, 2)
        pred_density = self.decoder(torch.cat((corr, surg_features), dim=1))

        # Save intermediate results for logging
        if set_img_dict:
            similarity = torch.sum(corr, dim=1, keepdim=True)
            self.img_dict.update({
                'img': imgs[0],
                'pred': pred_density[0],
                'sim': similarity[0],
            })

        return pred_density

    def forward(self, inp_dict, set_img_dict=False):
        img, class_name = inp_dict['img'].cuda(), inp_dict['class_name']
        pred_density_map = self.predict(img, class_name, set_img_dict)

        # Calculate loss
        metric_dict = {}
        gt = inp_dict['gt']
        l2_density_loss = self.mse_loss(gt, pred_density_map)
        total_loss = l2_density_loss

        if self.simmap_ce_lambda:
            large_sim = F.interpolate(self.output_dict_for_loss['sim'],
                                      (self.args.input_resolution, self.args.input_resolution),
                                      align_corners=False, mode='bilinear')
            ce_loss = -torch.mean(torch.log(torch.clamp(large_sim, min=1e-6)) * gt)
            total_loss = total_loss + ce_loss
            metric_dict.update({
                'ce_loss': ce_loss.item()
            })

        metric_dict.update({
            'l2_density': l2_density_loss.item(),
            'total_loss': total_loss.item(),
        })

        if set_img_dict:
            self.img_dict['class_name'] = inp_dict['class_name'][0]
            self.img_dict['gt'] = gt[0]
            self.img_dict['count'] = inp_dict['count'][0]

        return total_loss, metric_dict


class CLIPSCorrCNNv2(BaseModel):
    """
        Use surgery ViT feature
    """
    def __init__(self, args):
        super(CLIPSCorrCNNv2, self).__init__(args=args)
        self.txt_backbone = build_txt_encoder(args)
        self.img_backbone = build_img_encoder(args)

        self.decoder = CNNDecoderwCNNFeat(in_dim=args.txt_emb_dim)
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
        B, _, H, W = imgs.size()
        npH, npW = H // self.args.patch_size, W // self.args.patch_size
        _, surg_features, _ = self.img_backbone(imgs)  # [B, 1, 512], [B, 1024, 512]

        # Normalization
        txt_embeddings = txt_embeddings / txt_embeddings.norm(dim=-1, keepdim=True)
        surg_features = surg_features / surg_features.norm(dim=-1, keepdim=True)

        corr = torch.mul(txt_embeddings.unsqueeze(1), surg_features)
        corr = corr.reshape(B, npH, npW, -1).permute(0, 3, 1, 2)

        pred_density = self.decoder(imgs, corr)

        # Save intermediate results for logging
        if set_img_dict:
            similarity = torch.sum(corr, dim=1, keepdim=True)
            self.img_dict.update({
                'img': imgs[0],
                'pred': pred_density[0],
                'sim': similarity[0],
            })

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
