import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

def CNN(self):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        conv_layers = []
        in_channels = cfg.model.conv.in_channels               # nombre de channels à l'entrée
        for out in cfg.model.conv.out_channels:  # parcourt la liste des tailles des différents canaux
          conv_layers.append(
          nn.Conv2d(
                  in_channels=in_channels,
                  out_channels=out,
                  kernel_size=cfg.kernel_size,
                  stride=cfg.stride,
                  padding=cfg.padding
                ))
          conv_layers.append(instantiate(cfg.cnn_activation.activation)) # activation du CNN
          conv_layers.append(nn.MaxPool2d(cfg.model.conv.pool_size))
          in_channels = out
        self.cnn = nn.Sequential(*conv_layers)
        
    def forward(self, x):
        return self.cnn(x)