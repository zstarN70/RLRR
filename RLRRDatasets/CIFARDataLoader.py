import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar_loader(dataset_name, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data_cifar', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='./data_cifar', train=False, download=True, transform=test_transform)
    elif dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data_cifar', train=True, download=True,
                                                 transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=True,
                                                transform=test_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

    return train_loader, test_loader