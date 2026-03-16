import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

class MLP(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        mlp_layers = []
        mlp_dims = cfg.model.mlp_dims
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                mlp_layers.append(instantiate(cfg.mlp_activation.activation))
                if "dropout" in cfg.mlp_init:
                    mlp_layers.append(nn.Dropout(cfg.mlp_init.dropout))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp(x)