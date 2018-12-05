from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import img_process as process

mean = np.array([0.485, 0.406, 0.456])
std = np.array([0.229, 0.224, 0.225])

def load_training(root_path, txt_path, batch_size, kwargs):

    train_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomCrop((224, 224), padding=7), transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    #train_transform = transforms.Compose(
    #    [transforms.RandomHorizontalFlip(), transforms.RandomCrop((224, 224), padding=7), transforms.ToTensor()])

    #train_transform = transforms.Compose(
    #    [transforms.Resize([256, 256]),
    #     transforms.RandomCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()])
    txt_path = os.path.join(root_path, txt_path)
    data = process.customData(txt_path = txt_path, data_transforms=train_transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader, data.return_num_class()

def load_testing(root_path, txt_path, batch_size, kwargs):

    test_trainsform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(), transforms.Normalize(mean, std)])
    txt_path = os.path.join(root_path, txt_path)

    data = process.customData(txt_path = txt_path, data_transforms=test_trainsform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader, data.return_num_class()