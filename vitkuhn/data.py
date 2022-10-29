# torch dataset image folder

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision import transforms

# 加上transforms
normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
transform = transforms.Compose([
    transforms.RandomCrop(180),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    normalize
])


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super().__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


if __name__ == '__main__':
    # create the dataset and dataloader
    dataset = torchvision.datasets.ImageFolder(root=r'd:\ANewspace\code\vit-pytorch\datasets\num345', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # helper function to show an image
    # (used in the `plot_classes_preds` function below)
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    print("--------------------------------------------------")
