import torch
import os
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

def download_minst():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)


if __name__ == "__main__":
    download_minst()
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Visualize some samples
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    plt.imshow(images[0][0], cmap='gray')
    plt.title(f'Label: {labels[0]}')
    plt.show()