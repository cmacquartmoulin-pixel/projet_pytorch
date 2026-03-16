from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torchinfo import summary

def init_weights(module, init_cfg, activation_name): 
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if init_cfg.type.lower() == "kaiming":
            nn.init.kaiming_normal_(
                module.weight,
                mode=init_cfg.get("mode", "fan_in"), #remplace init_cfg.mode mais si mode n'existe pas, on a une valeur par défaut.
                nonlinearity=activation_name.lower()
            )
        elif init_cfg.type.lower() == "xavier":
            nn.init.xavier_normal_(
                module.weight,
                gain=init_cfg.get("gain", 1.0)                
            )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def Net(self):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        # CNN
        conv_layers = []
        in_channels = cfg.model.conv.in_channels               # nombre de channels à l'entrée
        for out in cfg.model.conv.out_channels:  # parcourt la liste des tailles des différents canaux
          conv_layers.append(
          nn.Conv2d(
                  in_channels=in_channels,
                  out_channels=out,
                  kernel_size=cfg.kernel_size,
                  stride=cgf.stride,
                  padding=cgf.padding
                ))
          conv_layers.append(instantiate(cfg.cnn_activation.activation)) # activation du CNN
          conv_layers.append(nn.MaxPool2d(cfg.model.conv.pool_size))
          in_channels = out
        self.cnn = nn.Sequential(*conv_layers)

        # MLP



    def forward(self, x):
        x = self.cnn(x)
        print("After CNN:", x.shape)
        x = self.flatten(x)
        print("After Flatten:", x.shape)
        x = self.mlp(x)
        print("After MLP:", x.shape)
        return x