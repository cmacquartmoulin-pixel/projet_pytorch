import torch
from models import Net
from dataset import get_dataloaders
import hydra
from omegaconf import DictConfig

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modele
    model = Net(cfg).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.eval()

    # Charger le testloader
    _, testloader = get_dataloaders(cfg)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    print(f"Accuracy : {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()