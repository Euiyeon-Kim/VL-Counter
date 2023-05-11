import os
import json
import random

import cv2
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


TTensor = transforms.Compose([
    transforms.ToTensor(),
])

Augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.GaussianBlur(kernel_size=(7, 9))
])


class FSC147(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self.data_root = args.data_root
        self.density_scale = args.density_scale
        self.input_resolution = args.input_resolution

        self.img_root = os.path.join(self.data_root, "images_384_VarV2")
        self.gt_root = os.path.join(self.data_root, "gt_density_map_adaptive_384_VarV2")

        assert os.path.exists(self.img_root), f'{self.img_root} should exists'
        assert os.path.exists(self.gt_root), f'{self.gt_root} should exists'

        data_split_file = os.path.join(self.data_root, "Train_Test_Val_FSC_147.json")
        with open(data_split_file) as f:
            data_split = json.load(f)
        img_ids = data_split[mode]

        image_classes_file = os.path.join(self.data_root, 'ImageClasses_FSC147.txt')
        img_classes = {}
        with open(image_classes_file) as f:
            for line in f.readlines():
                l = line.strip()
                if len(l) != 0:
                    im_id, class_name = l.split('\t')
                    img_classes[im_id] = class_name

        self.datas = [(img_classes[img_id], img_id) for img_id in img_ids]
        self.n_samples = len(self.datas)

        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=args.img_norm_mean, std=args.img_norm_var)
        ])

    def __len__(self):
        return self.n_samples

    def augment(self, img, density):
        resized_image = transforms.Resize((self.input_resolution, self.input_resolution),
                                interpolation=transforms.InterpolationMode.BICUBIC)(img)
        resized_density = cv2.resize(density, (self.input_resolution, self.input_resolution))
        orig_count = np.sum(density)
        new_count = np.sum(resized_density)
        if new_count > 0:
            resized_density = resized_density * (orig_count / new_count)

        # Gaussian noise
        resized_image = TTensor(resized_image)
        if self.mode != 'train':
            resized_density = resized_density * self.density_scale
            return resized_image, resized_density, new_count

        # Augmentation probability
        aug_p = random.random()
        aug_flag = 1
        if aug_p < 0.4:  # 0.4
            aug_flag = 1
            if aug_p < 0.25:  # 0.25
                aug_flag = 0
                
        if aug_flag == 1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            resized_image = resized_image + noise
            resized_image = torch.clamp(resized_image, 0, 1)

        # Color jitter and Gaussian blur
        if aug_flag == 1:
            resized_image = Augmentation(resized_image)

        # Random horizontal flip
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                resized_image = TF.hflip(resized_image)
                resized_density = TF.hflip(Image.fromarray(resized_density))

        # Gaussian distribution density map
        resized_density = ndimage.gaussian_filter(resized_density, sigma=(1, 1), order=0)

        # Density map scale up
        resized_density = resized_density * self.density_scale
        return resized_image, resized_density, new_count

    def __getitem__(self, item):
        cur_class, img_file_name = self.datas[item]
        img_pth = os.path.join(self.img_root, img_file_name)
        gt_pth = os.path.join(self.gt_root, img_file_name.replace('.jpg', '.npy'))
        gt = np.load(gt_pth).astype(np.float32)
        img = Image.open(img_pth)

        img, gt, count = self.augment(img, gt)

        img = self.img_transform(img).float()
        gt = torch.from_numpy(gt).float().unsqueeze(0)

        return {
            'class_name': cur_class,
            'gt': gt,
            'count': count,
            'img': img,
        }
