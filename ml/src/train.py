import copy
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

from models.cnn import CNNNet, preprocess
from training.train import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

data_dir = 'res/datasets/math_symbols_small'
full_dataset = datasets.ImageFolder(data_dir, preprocess)
test_train_split = 0.9
train_size = int(test_train_split * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


dataloaders = {
    "train": torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
    "test":  torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
}
dataset_sizes = {
    "train": len(train_dataset),
    "test": len(test_dataset)
}

print(dataloaders)
print(dataset_sizes)

print(full_dataset.classes)

model = CNNNet(len(full_dataset.classes))
model = model.to(device)

train_model(model, criterion, dataloaders, dataset_sizes, device)
