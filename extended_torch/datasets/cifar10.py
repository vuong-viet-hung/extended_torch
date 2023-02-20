import torch.utils.data
import torchvision


from extended_torch import one_hot_encode


def create_transform(train: bool):
    transforms = [
        torchvision.transforms.ToTensor(),
    ]

    if train:
        transforms.extend(
            [
                torchvision.transforms.RandomResizedCrop((32, 32)),
                torchvision.transforms.RandomVerticalFlip(),
            ]
        )

    transforms.append(
        torchvision.transforms.Lambda(lambda image: image / 255)
    )
    return torchvision.transforms.Compose(transforms)


class CIFAR10Images(torch.utils.data.Dataset):

    def __init__(self, root: str, train: bool) -> None:
        self._dataset = torchvision.datasets.CIFAR10(
            root, train, transform=create_transform(train), download=True
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx) -> torch.Tensor:
        image, _ = self._dataset[idx]
        return image


class CIFAR10(torch.utils.data.Dataset):

    def __init__(self, root: str, train: bool) -> None:
        self._dataset = torchvision.datasets.CIFAR10(
            root, train, create_transform(train), download=True
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._dataset[idx]
        return image, one_hot_encode(
            torch.tensor(label).type(torch.long), n_classes=10
        )
