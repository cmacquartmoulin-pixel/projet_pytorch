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


"""    
import optuna
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import wandb

from models import Net
from dataset import get_dataloaders


# ---------------------------------------------------------------
# Build a config from Optuna suggestions (overrides the yaml)
# ---------------------------------------------------------------
def build_cfg(trial: optuna.Trial, base_cfg: DictConfig) -> DictConfig:
    cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)

    # --- optimizer: lr ---
    cfg_dict["optimizer"]["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # --- scheduler: step_size and gamma ---
    cfg_dict["scheduler"]["step_size"] = trial.suggest_int("step_size", 3, 10)
    cfg_dict["scheduler"]["gamma"]     = trial.suggest_float("gamma", 0.1, 0.7)

    # --- batch_size ---
    cfg_dict["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])

    # --- MLP architecture ---
    # We fix input (4096) and output (10) and let Optuna choose hidden layers
    hidden1  = trial.suggest_categorical("hidden1",  [256, 512, 1024])
    hidden2  = trial.suggest_categorical("hidden2",  [64, 128, 256])
    dropout  = trial.suggest_float("dropout", 0.1, 0.5)
    cfg_dict["model"]["mlp"]["mlp_dims"] = [4096, hidden1, hidden2, 10]
    cfg_dict["model"]["mlp"]["dropout"]  = dropout

    return OmegaConf.create(cfg_dict)


# ---------------------------------------------------------------
# One Optuna trial = one full training run
# ---------------------------------------------------------------
def objective(trial: optuna.Trial, base_cfg: DictConfig) -> float:
    cfg    = build_cfg(trial, base_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(base_cfg.seed)

    # Data (uses the new batch_size from cfg)
    train_loader, val_loader = get_dataloaders(cfg)

    # Model
    model     = Net(cfg).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    scheduler = StepLR(optimizer, step_size=cfg.scheduler.step_size,
                       gamma=cfg.scheduler.gamma)

    # --- wandb (one run per trial, grouped under the study) ---
    run_name = f"trial_{trial.number}"
    wandb.init(
        project  = "cifar-project",
        entity   = "anateresafalcao-t-l-com-physique-strasbourg",
        name     = run_name,
        group    = "optuna_search",
        config   = OmegaConf.to_container(cfg, resolve=True),
        reinit   = True,
    )

    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        # ---- train ----
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y    = x.to(device), y.to(device)
                out     = model(x)
                val_loss += loss_fn(out, y).item()
                _, pred  = out.max(1)
                correct += pred.eq(y).sum().item()
                total   += y.size(0)

        val_loss /= len(val_loader)
        val_acc   = 100.0 * correct / total

        wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc})

        # Optuna pruning: stop unpromising trials early
        trial.report(val_loss, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    wandb.finish()
    return best_val_loss   # Optuna minimizes this value


# ---------------------------------------------------------------
# Main: load base config, run the study
# ---------------------------------------------------------------
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    # Load the same config.yaml used in main.py (without @hydra.main decorator)
    with hydra.initialize(config_path="../conf", version_base=None):
        base_cfg = hydra.compose(config_name="config")

    # Pruner: cuts trials that are clearly worse than others early
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    study   = optuna.create_study(
        direction    = "minimize",   # we want to minimize val_loss
        pruner       = pruner,
        study_name   = f"cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        storage      = "sqlite:///optuna_cifar10.db",  # saves results to a local DB
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg),
        n_trials  = 20,     # number of combinations to try
        timeout   = 3600,   # max 1 hour (optional safety limit)
    )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------
    print("\n========== BEST TRIAL ==========")
    best = study.best_trial
    print(f"  Val loss : {best.value:.4f}")
    print("  Params   :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
        """"

    # Tip: visualize all trials with:
    # optuna-dashboard sqlite:///optuna_cifar10.db