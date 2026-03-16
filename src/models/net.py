# assembly CNN and MLP, initilize weights

import torch.nn as nn
from omegaconf import DictConfig
from .cnn import CNN
from .mlp import MLP

class Net(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cnn    = CNN(cfg)
        self.flatten = nn.Flatten()
        self.mlp    = MLP(cfg)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(
                    m,
                    cfg.cnn_init,
                    cfg.cnn_activation.activation._target_.split(".")[-1]
                )
            elif isinstance(m, nn.Linear):
                init_weights(
                    m,
                    cfg.mlp_init,
                    cfg.mlp_activation.activation._target_.split(".")[-1]
                )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x
    
def init_weights(module, init_cfg, activation_name):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if init_cfg.type.lower() == "kaiming":
            nn.init.kaiming_normal_(
                module.weight,
                mode=init_cfg.get("mode", "fan_in"),
                nonlinearity=activation_name.lower()
            )
        elif init_cfg.type.lower() == "xavier":
            nn.init.xavier_normal_(
                module.weight,
                gain=init_cfg.get("gain", 1.0)
            )
        if module.bias is not None:
            nn.init.zeros_(module.bias)