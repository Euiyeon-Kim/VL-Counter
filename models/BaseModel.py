from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args

    @abstractmethod
    def forward(self, *args, **kwargs):
        return

    @abstractmethod
    def inference(self, *args, **kwargs):
        return

    @abstractmethod
    def get_img_log_dict(self, *args, **kwargs):
        return


