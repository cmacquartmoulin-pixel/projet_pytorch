import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

class CNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        conv_layers = []
        in_channels = cfg.model.cnn.in_channels               # nombre de channels à l'entrée
        for out in cfg.model.cnn.out_channels:  # parcourt la liste des tailles des différents canaux
          conv_layers.append(
          nn.Conv2d(
                  in_channels=in_channels,
                  out_channels=out,
                  kernel_size=cfg.model.cnn.kernel_size,  
                  stride=cfg.model.cnn.stride,           
                  padding=cfg.model.cnn.padding 
                ))
          conv_layers.append(instantiate(cfg.cnn_activation.activation)) # activation du CNN
          conv_layers.append(nn.MaxPool2d(cfg.model.cnn.pool))
          in_channels = out
        self.cnn = nn.Sequential(*conv_layers)
        
    def forward(self, x):
        return self.cnn(x)