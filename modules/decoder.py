import torch
import torch.nn as nn

from .small_encoder import Encoder


class DensityX16(nn.Module):
    def __init__(self, in_dim):
        super(DensityX16, self).__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_dim, 196, 7, padding=3),
            nn.PReLU(196),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.PReLU(128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.PReLU(64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.PReLU(32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 1, 1),
        )
        self._weight_init_()

    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pred = self.regressor(x)
        return torch.clamp(pred, min=0)


class CNNDecoderwCNNFeat(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.visual_encoder = Encoder()
        self.density_regressor = DensityX16(in_dim=in_dim + 96)

    def forward(self, img, sim_map):
        img_feat = self.visual_encoder(img)
        density = self.density_regressor(torch.cat((sim_map, img_feat), dim=1))
        return density
