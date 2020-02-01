import random

import torch
from torchvision import datasets

import scripts.models.mobilenet as mm

from torch.utils.data import DataLoader


def valid_func(x):
    return random.random() < 0.1


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = 'res/test'
    full_dataset = datasets.ImageFolder(data_dir, mm.preprocess)
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True,
                            num_workers=4)

    state_dict = torch.load("res/mobile_cnn.pth", map_location=torch.device('cpu'))
    model = mm.MobileNet(features=state_dict['mobilenet.classifier.1.bias'].shape[0], pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device)

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        print(preds, full_dataset.classes[labels], outputs)
