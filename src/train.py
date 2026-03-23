from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf 
import torch
import torch.nn as nn
from torchinfo import summary
from datetime import datetime
import wandb

from models import Net
from dataset import get_dataloaders

class EarlyStopping:
    def __init__(self, cfg):
        self.enabled   = cfg.early_stopping.enabled
        self.patience  = cfg.early_stopping.patience
        self.min_delta = cfg.early_stopping.min_delta
        self.counter   = 0
        self.best_loss = float('inf')
        self.stop      = False

    def step(self, val_loss):
        if not self.enabled:
            return
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss  # better → resets
            self.counter   = 0
        else:
            self.counter += 1          # not better → increases
            if self.counter >= self.patience:
                self.stop = True


# fonction avec boucle sur toutes les époques
def train(model, optimizer, loss, scheduler, train_loader, val_loader, device, cfg):
    early_stopping = EarlyStopping(cfg)  # inicializes 

    for epoch in range(cfg.epochs):
        train_loss, train_acc = _train_epoch(model, optimizer, loss, train_loader, device)
        val_loss, val_acc   = _val_epoch(model, loss, val_loader, device)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        })


        early_stopping.step(val_loss)    # verifies if it should stop
        
        if early_stopping.stop:
            print(f"Early stopping for epoch {epoch+1}")
            break
        
    torch.save(model.state_dict(), "model.pth")
    wandb.save("model.pth")
    print("Modele sauvegardé dans model.pth")
    


# fonction d'apprentissage : poids modifiés
def _train_epoch(model, optimizer, loss, train_loader, device):
    model.train()
    total_loss = 0
    correct = 0  # number of correctly classified images
    total = 0    # number of total images seen so far
    
    for x, y in train_loader:               # each iteration we take the next batch
        # a batch is a tuple of two tensors, the 1st: images, 2nd: labels
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()               # reset gradients to 0
        outputs = model(x)
        output_loss = loss(outputs, y)      # compute the errors
        output_loss.backward()              # compute the gradients
        optimizer.step()                    # update the weights
        total_loss += output_loss.item()    # accumulate the loss
        
        _, predicted = outputs.max(1)       # _ is a convention to a variable that won't be used (outputs.max(1) returns two values: the highest probability and the corresponding class)
        total   += y.size(0)
        correct += predicted.eq(y).sum().item()
        
    return total_loss / len(train_loader), 100 * correct / total # return the average loss and the accuracy


def _val_epoch(model, loss, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            output_loss = loss(outputs, y)
            total_loss += output_loss.item()
            
            _, predicted = outputs.max(1)
            total   += y.size(0)
            correct += predicted.eq(y).sum().item()   #compares the predicted labels with the true labels and counts how many are correct  
            
    return total_loss / len(val_loader), 100 * correct / total


@hydra.main(version_base=None, config_path="../conf", config_name="config")

def main(cfg: DictConfig):

    # Reproductibility
    torch.manual_seed(cfg.seed)

    # Device : Utilise le GPU si dispo, sinon le CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialisation de wandb
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="cifar-project",
        entity="anateresafalcao-t-l-com-physique-strasbourg",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Model
    model = Net(cfg).to(device) # Crée un réseau CNN+MLP et l'envoie sur le bon device
    # summary(model, input_size=(1, 3, 32, 32))  # Show informations about the network
    wandb.watch(model, log="gradients", log_freq=10)

    # Hydra lit les fichiers `.yaml` et crée automatiquement la **loss**, l'**optimiseur** et le **scheduler**.
    loss = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    # scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # Training
    train(model=model, optimizer=optimizer, loss=loss,
      scheduler=None, train_loader=train_loader,
      val_loader=val_loader, device=device, cfg=cfg)
    
    wandb.finish()


if __name__ == "__main__":
    main()