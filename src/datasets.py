
import torch
from PIL import Image
import matplotlib.pyplot as plt

import glob
from torch.functional import split
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import os

import torchvision

class MyDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        # self.to_tensor = transforms.ToTensor()
        self.transforms =transforms.Compose([
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

        images = os.listdir(main_dir)
        self.total_images = images

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, index) :
        image_loc = os.path.join(self.main_dir,self.total_images[index])
        image =  Image.open(image_loc)
        image = self.transforms(image)
        return image  


def get_dataset(dataroot, image_size, batch_size, num_workers):
    # We can use an image folder dataset the way we have it setup.
    dataset = torchvision.datasets.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    return dataloader

