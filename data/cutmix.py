import numpy as np

import torch



def cutmix(batch, alpha):
    img = batch['img']
    gt = batch['gt']
    count = batch['count']
    class_name = batch['class_name']
    
    B, _, H, W = img.shape
    indices = torch.randperm(B).long()
    shuffled_img = img[indices]
    shuffled_gt = gt[indices]
    shuffled_class_name = np.array(class_name)[indices]

    lam = np.random.beta(alpha, alpha)
    
    cx = np.random.uniform(0, W)
    cy = np.random.uniform(0, H)
    w = W * np.sqrt(1 - lam)
    h = H * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, W)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, H)))

    # CutMix image
    img[:, :, y0:y1, x0:x1] = shuffled_img[:, :, y0:y1, x0:x1]

    
    # CutMix GT
    same_idx = torch.tensor(class_name == shuffled_class_name).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    gt[:,:,y0:y1,x0:x1] = 0
    cutmix_gt = torch.zeros_like(shuffled_gt)
    cutmix_gt[:,:,y0:y1,x0:x1] = shuffled_gt[:,:,y0:y1,x0:x1]
    shuffled_targets = gt + same_idx * cutmix_gt
    
    return  {
        'img': img,
        'gt': shuffled_targets,
        'count': count,
        'class_name': class_name,
    }
    

class CutMixCollator:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch
