import os
import torch
import torch.utils.data as data
from PIL import Image


class AzureMNIST(data.Dataset):
    def __init__(self, data_folder, train=False, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        if train:
            data_file = "training.pt"
        else:
            data_file = "test.pt"

        self.data, self.targets = torch.load(os.path.join(data_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
