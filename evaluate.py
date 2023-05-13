import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import data
from utils.visualize import denormalize, save_density_map_w_similarity, scale_and_get_colormap


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


@torch.no_grad()
def test_fsc384(args, model, batch_size=8, return_visual=False):
    model.eval()
    mae = 0
    rmse = 0
    img_dict = {}
    test_dataset = data.FSC147(args, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8,
                                pin_memory=True, shuffle=True, drop_last=False)
    print('Number of test images: %d' % len(test_dataset))

    for _, batch in enumerate(tqdm(test_dataloader)):
        img = batch['img'].cuda()
        class_name = batch['class_name']
        gt_cnt = batch['count'].cuda()
        pred = model.inference(img, class_name, set_img_dict=return_visual)

        pred_cnt = torch.sum(pred, dim=(1, 2, 3)) / args.density_scale
        cnt_err = abs(pred_cnt - gt_cnt)
        mae += torch.sum(cnt_err).item()
        rmse += torch.sum(cnt_err**2).item()

    mae = mae / len(test_dataset)
    rmse = (rmse / len(test_dataset)) ** 0.5
    print(f"Current MAE: {mae} \t RMSE: {rmse}")
    return {
        'test/fsc_mae': mae,
        'test/fsc_rmse': rmse,
    }, img_dict


@torch.no_grad()
def test_prompt(args, model, img_path, test_classes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(img_path)
    resized_image = transforms.Resize((args.input_resolution, args.input_resolution),
                                      interpolation=transforms.InterpolationMode.BICUBIC)(img)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.img_norm_mean, std=args.img_norm_var)
    ])
    inp_img = img_transform(resized_image).unsqueeze(0).to(DEVICE)
    visualize = []

    for class_name in test_classes:
        pred = model.inference(inp_img, [class_name], set_img_dict=True)
        pred_cnt = torch.sum(pred) / args.density_scale
        viz_density = scale_and_get_colormap(pred[0].permute(1, 2, 0).cpu().numpy())
        cv2.putText(viz_density, class_name, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cv2.putText(viz_density, f"{pred_cnt:.5f}", (10, 512 - 15), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)

        log_img_dict = model.img_dict
        sim = F.interpolate(log_img_dict['sim'].unsqueeze(1), size=(512, 512), mode='nearest')[0] \
            .permute(1, 2, 0).cpu().numpy()
        viz_sim = scale_and_get_colormap(sim)

        origin_sim = F.interpolate(log_img_dict['origin_sim'].unsqueeze(1), size=(512, 512), mode='nearest')[0] \
            .permute(1, 2, 0).cpu().numpy()
        viz_origin_sim = scale_and_get_colormap(origin_sim)

        viz = np.hstack((viz_density, viz_origin_sim, viz_sim))
        visualize.append(viz)

    viz = np.vstack((visualize)).astype(np.uint8)
    viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
    img_name = img_path.split("/")[-1]
    img_line = np.zeros((args.input_resolution*len(test_classes), args.input_resolution, 3)).astype(np.uint8)
    img_line[:args.input_resolution, :args.input_resolution, :] = resized_image
    Image.fromarray(np.hstack((img_line, viz))).save(f"{save_dir}/{img_name}")


if __name__ == '__main__':
    PTH_NAME = 'best_mae'
    EXP_NAME = 'CLIPCorrCNN/fixCLIP_allNorm_CNNFeat_noReLU'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import yaml
    from dotmap import DotMap

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
    model.load_state_dict(ckpt, strict=True)
    
    for param in model.parameters():
        param.requires_grad = False

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    
    # IMG_PATH = 'datasets/FSC147_384_V2/images_384_VarV2/343.jpg'
    IMG_PATH = 'banana_nuts.jpeg'
    CLASSES = ['nut', 'nuts', 'banana', 'bananas']
    test_prompt(args, model, IMG_PATH, CLASSES, save_dir=f"exps/{EXP_NAME}/testing")
    # validate_fsc384(args, model, batch_size=8)
    # test_fsc384(args, model, batch_size=8)
