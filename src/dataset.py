import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from omegaconf import DictConfig

AUGMENTATION_REGISTRY = {
    "horizontal_flip": lambda aug: transforms.RandomHorizontalFlip(),
    "random_crop":     lambda aug: transforms.RandomCrop(aug.size, padding=aug.padding),
}

def get_dataloaders(cfg: DictConfig):
    train_transforms_list = []

    for name, aug in cfg.augmentation.items():
        if not isinstance(aug, str) and not aug.get("enabled", True):
            continue
        if name in AUGMENTATION_REGISTRY:
            train_transforms_list.append(AUGMENTATION_REGISTRY[name](aug))
        else:
            print(f"[WARNING] '{name}' not found in registry, ignored.")

    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
    train_transforms_list.append(transforms.ToTensor())
    train_transforms_list.append(normalize)
    
    # Val/test: só normaliza, sem augmentation
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_transform = transforms.Compose(train_transforms_list)
    

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=val_transform)
    
    if cfg.debug:
        trainset = Subset(trainset, range(cfg.debug_size))
        testset  = Subset(testset,  range(cfg.debug_size // 5))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader

"""""
#TEST DU DEBUT 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
"""