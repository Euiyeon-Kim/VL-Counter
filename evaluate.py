import os

from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import data
from utils.visualize import denormalize, save_density_map_w_similarity


@torch.no_grad()
def validate_fsc384(args, model, batch_size=8, return_visual=True):
    model.eval()
    mae = 0
    rmse = 0
    img_dict = {}
    val_dataset = data.FSC147(args, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,
                                pin_memory=True, shuffle=True, drop_last=False)
    print('Number of validation images: %d' % len(val_dataset))

    for _, batch in enumerate(tqdm(val_dataloader)):
        img = batch['img'].cuda()
        class_name = batch['class_name']
        gt_cnt = batch['count'].cuda()
        pred = model.inference(img, class_name, set_img_dict=return_visual)

        pred_cnt = torch.sum(pred, dim=(1, 2, 3)) / args.density_scale
        cnt_err = abs(pred_cnt - gt_cnt)
        mae += torch.sum(cnt_err).item()
        rmse += torch.sum(cnt_err**2).item()

    if return_visual:
        log_img_dict = model.img_dict
        denormed = denormalize(log_img_dict['img'].unsqueeze(0), args.img_norm_mean, args.img_norm_var)
        img = save_density_map_w_similarity(denormed, log_img_dict['pred'], log_img_dict['sim'],
                                            log_img_dict['origin_sim'], batch['gt'][0], gt_cnt[0],
                                            class_name[0], args.density_scale)
        img_dict = {'val/fsc_pred': torch.from_numpy(img).permute(2, 0, 1) / 255.}

    mae = mae / len(val_dataset)
    rmse = (rmse / len(val_dataset)) ** 0.5
    print(f"Current MAE: {mae} \t RMSE: {rmse}")
    return {
        'val/fsc_mae': mae,
        'val/fsc_rmse': rmse,
    }, img_dict



if __name__ == '__main__':
    PTH_NAME = 'best_mae'
    EXP_NAME = 'CNNSlotCorr/V1_res18L3_dim256_slot3_iter5_noAbs_b16'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import yaml
    from dotmap import DotMap
    from torch.nn.parallel import DistributedDataParallel as DDP

    import models

    # Parse argument
    config_path = f"exps/{EXP_NAME}/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = DotMap(config)

    # Define model
    model = getattr(models, args.model)(args).to(DEVICE)

    # Load weights
    ckpt_path = f"exps/{EXP_NAME}/{PTH_NAME}.pth"
    ckpt = torch.load(ckpt_path)['model']
    model_ckpt = {}
    for k, v in ckpt.items():
        model_ckpt[k[7:]] = v
    model.load_state_dict(model_ckpt, strict=True)

    for param in model.parameters():
        param.requires_grad = False

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
