import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

class CNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        conv_layers = []     # Initialisation d'une liste vide qui va accumuler les couches
        in_channels = cfg.model.cnn.in_channels  # nombre de channels à l'entrée (1 pour grayscale, 3 pour RGB)
        for out in cfg.model.cnn.out_channels:  # on itère sur la liste des canaux de sortie définie dans la config
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