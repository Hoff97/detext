import io
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations import Compose, ElasticTransform, ShiftScaleRotate
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import detext.server.ml.models.mobilenet as mm
from detext.server.ml.training.balanced_ds import BalancedDS
from detext.server.ml.training.dataloader import DBDataset
from detext.server.ml.training.train import train_model
from detext.server.ml.util.util import augment_image, eval_model
from detext.server.models import MathSymbol, TrainImage


def valid_func(x):
    return random.random() < 2


def get_data(item):
    image = Image.open(io.BytesIO(item.image))

    return augment_image(image)


def get_label(item):
    return item.symbol.name


def get_class_name(item):
    return item.name


def run(num_epochs=1, device="cuda"):
    criterion = nn.CrossEntropyLoss()

    full_dataset = DBDataset(TrainImage, MathSymbol, get_data, get_label,
                             get_class_name, filter=valid_func)

    full_dataset = BalancedDS(full_dataset, min_count=100)

    test_train_split = 0.9
    train_size = int(test_train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset,
                                               [train_size, test_size])

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=32, num_workers=1),
        "test":  DataLoader(test_dataset, batch_size=16, shuffle=True,
                            num_workers=1)
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "test": len(test_dataset)
    }

    print(dataloaders)
    print(dataset_sizes)

    model = mm.MobileNet(features=len(full_dataset.classes), pretrained=True)
    model = model.to(device)

    model, accuracy = train_model(model, criterion, dataloaders, device,
                                  num_epochs=num_epochs)

    model = model.to('cpu')
    torch.save(model.state_dict(), "test_augment_2.pth")

    model = model.to('cuda')
    eval_model(model, dataloaders["test"], "cuda", len(full_dataset.classes))
