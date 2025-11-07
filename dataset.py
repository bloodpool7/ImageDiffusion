
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10Dataset:
    def __init__(self, batch_size=4, num_workers=2, prefetch_factor=2, dataset_path='./data'):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             torchvision.transforms.RandomHorizontalFlip(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        full_trainset = torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=True,
            download=True,
            transform=self.transform,
        )

        train_len, val_len = 45000, 5000
        self.trainset, self.valset = torch.utils.data.random_split(
            full_trainset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            multiprocessing_context='spawn',
        )

        self.valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                               download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def get_trainloader(self):
        return self.trainloader

    def get_testloader(self):
        return self.testloader

    def get_valloader(self):
        return self.valloader

    def get_classes(self):
        return self.classes


class FlowersDataset:
    def __init__(self,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 dataset_path: str = './data/flowers',
                 image_size: int = 32
                 ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dir = Path(dataset_path) / 'train'
        if not train_dir.exists():
            # Some dumps may put class folders directly at dataset_path
            train_dir = Path(dataset_path)

        self.trainset = torchvision.datasets.ImageFolder(root=str(train_dir), transform=self.transform)

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            multiprocessing_context='spawn',
        )

        # If ImageFolder, extract class names; else, generic
        if hasattr(self.trainset, 'classes'):
            self.classes = tuple(self.trainset.classes)
        else:
            self.classes = tuple()

    def get_trainloader(self):
        return self.trainloader

    def get_testloader(self):
        return None

    def get_valloader(self):
        return None

    def get_classes(self):
        return self.classes


if __name__ == "__main__":
    dataset = FlowersDataset()
    loader = dataset.get_trainloader()

    # Get some random training images
    dataiter = iter(loader)
    images, labels = next(dataiter)

    import utils
    utils.save_images(images, show=True)