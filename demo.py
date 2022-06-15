import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import Trainer
from test import Tester

from transformer import VisionTransformer


def main():

    # model parameters
    img_size = 28
    channels = 1
    patch_size = 4
    classes = 10
    embed_size = 128
    heads = 8
    dropout = 0
    fwd_expansion = 4
    layers = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # training parameters
    epochs = 32
    batch_size = 12
    lr = 0.01
    momentum = 0.05

    # network
    vit = VisionTransformer(
            img_size,
            channels,
            patch_size,
            classes,
            embed_size,
            heads,
            dropout,
            fwd_expansion,
            layers,
            device
            )

    # optimizer
    optimizer = optim.SGD(vit.parameters(), lr=lr, momentum=momentum)
    
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                'data/',
                train=True,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.1307),(0.3081,)
                        )
                    ])),
            batch_size=batch_size,
            shuffle=True
            )
    
    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                'data/',
                train=False,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.1307),(0.3081,)
                        )
                    ])),
            batch_size=batch_size,
            shuffle=True
            )
    
    train_dataset = tqdm(enumerate(train_loader),total=len(train_loader))
    trainer = Trainer(vit, optimizer, train_dataset)
        
    test_dataset = tqdm(enumerate(test_loader),total=len(test_loader))
    tester = Tester(vit, test_dataset)

    last_avg = 10
    for epoch in range(epochs):
        # training
        train_losses = trainer.train()
        
        # testing
        test_losses = tester.test()
        
        # early stopping
        test_avg = sum(test_losses)/len(test_losses)
        if(test_avg > last_avg):
            break
        last_avg = test_avg
        
   #     torch.save(vit.state_dict(), f'out/checkpoint_{epoch}.pth')



if __name__ == "__main__":
    main()

