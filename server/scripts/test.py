import random

import torch
import torch.nn as nn
from torchvision import datasets

import models.mobilenet as mm

from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()


def valid_func(x):
    return random.random() < 0.1


data_dir = 'res/test'
full_dataset = datasets.ImageFolder(data_dir, mm.preprocess)
dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True, num_workers=4)
model = torch.load("res/mobile_cnn.pth", map_location=torch.device('cpu'))
model.eval()
model = model.to(device)

for i, data in enumerate(dataloader):
    inputs, labels = data
    inputs = inputs.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    print(preds, full_dataset.classes[labels], outputs)
