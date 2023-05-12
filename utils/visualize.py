import cv2
import numpy as np

import torch
import torch.nn.functional as F


def denormalize(x, mean, std):
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def scale_and_get_colormap(x):
    x_0to255 = 255.0 * (x - np.min(x) + 1e-10) / (1e-10 + np.max(x) - np.min(x))
    viz_x = cv2.applyColorMap(x_0to255.astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_VIRIDIS)
    return viz_x


def save_density_map_w_similarity(denormed_img, pred_density, similarity, origin_similarity,
                                  gt, gt_count, class_chosen, density_scale):
    _, _, h, w = denormed_img.shape
    img = (denormed_img[0].permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)
    sim = F.interpolate(similarity.unsqueeze(1), size=(512, 512), mode='nearest')[0]\
        .permute(1, 2, 0).detach().cpu().numpy()
    viz_sim = scale_and_get_colormap(sim)

    origin_sim = F.interpolate(origin_similarity.unsqueeze(1), size=(512, 512), mode='nearest')[0]\
        .permute(1, 2, 0).detach().cpu().numpy()
    viz_origin_sim = scale_and_get_colormap(origin_sim)

    pred = pred_density.permute(1, 2, 0).detach().cpu().numpy() / density_scale
    pred_cnt = np.sum(pred)
    viz_pred = scale_and_get_colormap(pred)

    gt = gt.permute(1, 2, 0).cpu().numpy() / density_scale
    viz_gt = scale_and_get_colormap(gt)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(img, class_chosen, (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 1)

    viz_pred = viz_pred * 0.8 + img * 0.2
    cv2.putText(viz_pred, "Den Predict", (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)
    cv2.putText(viz_pred, f"{pred_cnt:.5f}", (0, h - 15), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)

    viz_gt = viz_gt * 0.8 + img * 0.2
    cv2.putText(viz_gt, "GT Count", (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)
    cv2.putText(viz_gt, f"{gt_count.item():.5f}", (0, h - 15), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)

    viz = np.hstack((img, viz_pred, viz_origin_sim, viz_sim, viz_gt)).astype(np.uint8)
    viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
    return viz


def save_density_map_without_gt(denormed_img, pred_density, class_chosen):
    _, h, w = denormed_img.shape
    img = (denormed_img.permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)

    denorm_pred = pred_density.permute(1, 2, 0).detach().cpu().numpy()
    pred_cnt = np.sum(denorm_pred)
    denorm_pred = 255.0 * (denorm_pred - np.min(denorm_pred) + 1e-10) / (
                1e-10 + np.max(denorm_pred) - np.min(denorm_pred))

    viz_pred = cv2.applyColorMap(denorm_pred.astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_VIRIDIS)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(img, class_chosen, (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 1)

    viz_pred = viz_pred * 0.8 + img * 0.2
    cv2.putText(viz_pred, "Den Predict", (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)
    cv2.putText(viz_pred, f"{pred_cnt:.5f}", (0, h - 15), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)

    viz = np.hstack((img, viz_pred)).astype(np.uint8)
    viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
    return viz
