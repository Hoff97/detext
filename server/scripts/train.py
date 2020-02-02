import io
import math
import random

import torch
import torch.nn as nn
from django.utils import timezone
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data.sampler import WeightedRandomSampler

import scripts.models.mobilenet as mm
from detext.server.models import ClassificationModel, MathSymbol, TrainImage
from scripts.training.dataloader import DBDataset
from scripts.training.train import train_model

from torch.utils.data import DataLoader


def valid_func(x):
    return random.random() < 2


def get_data(item):
    image = Image.open(io.BytesIO(item.image))
    data = mm.preprocess(image)
    return data.repeat((3, 1, 1))


def get_label(item):
    return item.symbol.name


def get_class_name(item):
    return item.name


def run(num_epochs=5, device="cuda"):
    criterion = nn.CrossEntropyLoss()

    full_dataset = DBDataset(TrainImage, MathSymbol, get_data, get_label,
                             get_class_name, filter=valid_func)
    test_train_split = 0.9
    train_size = int(test_train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset,
                                               [train_size, test_size])

    weights = torch.zeros(len(train_dataset))
    for i, data in enumerate(train_dataset):
        weights[i] = 1. / (math.log(full_dataset.class_counts[data[1]]) + 1.0)

    sampler = WeightedRandomSampler(weights, len(weights))

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=4, num_workers=1,
                            sampler=sampler),
        "test":  DataLoader(test_dataset, batch_size=4, shuffle=True,
                            num_workers=1)
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "test": len(test_dataset)
    }

    print(dataloaders)
    print(dataset_sizes)

    model = mm.MobileNet(features=len(full_dataset.classes), pretrained=False)
    model = model.to(device)

    model, accuracy = train_model(model, criterion, dataloaders, dataset_sizes,
                                  device, num_epochs=num_epochs)

    byteArr = io.BytesIO()
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, byteArr)

    torchByteArr = io.BytesIO()
    torch.save(model.state_dict(), torchByteArr)

    model_entity = ClassificationModel(None, model=byteArr.getvalue(),
                                       timestamp=timezone.now(),
                                       pytorch=torchByteArr.getvalue(),
                                       accuracy=accuracy)
    model_entity.save()
