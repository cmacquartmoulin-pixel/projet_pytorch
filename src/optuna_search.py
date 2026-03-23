import optuna
import torch
import torch.nn as nn
from hydra import initialize, compose
from omegaconf import OmegaConf

from models import Net
from dataset import get_dataloaders
from train import _train_epoch, _val_epoch


def objective(trial):
    # Charger la config Hydra de base
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")

    # Optuna suggère des valeurs pour ce trial
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True) # log=True signifie qu'il explore en échelle logarithmique 
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # On écrase les valeurs dans la config
    OmegaConf.update(cfg, "batch_size", batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(cfg)

    model = Net(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Entraînement court (5 époques suffit pour comparer)
    for epoch in range(5):
        _train_epoch(model, optimizer, loss_fn, train_loader, device)
        val_loss = _val_epoch(model, loss_fn, val_loader, device)

        # Optuna surveille et peut couper si c'est pas prometteur
        # Optuna vérifie si ce trial est mauvais comparé aux autres. 
        # Si oui, on lève une exception TrialPruned qui arrête l'essai proprement et passe au suivant. 
        # C'est le MedianPruner qui décide : il coupe si la val_loss est au-dessus de la médiane des trials précédents à la même époque.
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()  # coupe les trials mauvais
    )
    study.optimize(objective, n_trials=20)

    print("\n=== Meilleurs hyperparamètres ===")
    print(f" lr : {study.best_params['lr']:.6f}")
    print(f" batch_size : {study.best_params['batch_size']}")
    print(f" val_loss : {study.best_value:.4f}")